#!/usr/bin/env python3
"""Hermetic MODNet converter trust, graph and candidate transaction checks."""
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/models/convert_modnet_ncnn.py"
spec = importlib.util.spec_from_file_location("modnet_converter", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

GRAPH = "7767517\n3 3\nInput in0 0 1 in0\nReLU relu 1 1 in0 hidden\nSigmoid sigmoid 1 1 hidden out0\n"
# Real MODNet fuses sigmoid into Convolution; keep the fixture within that contract.
GRAPH = GRAPH.replace("Sigmoid sigmoid", "Convolution sigmoid")


class TrustAndGraphTests(unittest.TestCase):
    def test_promoted_metadata_and_license_reproduce_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            pack = first / "source" / converter.PACK_PATH
            pack.mkdir(parents=True)
            for name in ("model.onnx", "model.ncnn.param", "model.ncnn.bin"):
                (pack / name).write_bytes((name + " fixture").encode())
            summary = (ROOT / "resources/model_packs" / converter.PACK_PATH / "LICENSE.txt").read_text()
            license_text = converter.compose_license(summary, "Full upstream fixture license\n")
            (pack / "LICENSE.txt").write_text(license_text)
            converter.candidate_metadata(first, pack)
            # Emulate a later approved promotion, then export using its template.
            promoted = root / "promoted"
            template = promoted / "resources/model_packs" / converter.PACK_PATH
            template.parent.mkdir(parents=True)
            shutil.copytree(pack, template)
            catalog = promoted / "installer/models"
            catalog.mkdir(parents=True)
            shutil.copyfile(first / "curated-model-catalog-v1.candidate.json", catalog / "curated-model-catalog-v1.json")
            new_pack = second / "source" / converter.PACK_PATH
            shutil.copytree(pack, new_pack)
            with patch.object(converter, "ROOT", promoted):
                (new_pack / "LICENSE.txt").write_text(converter.compose_license((template / "LICENSE.txt").read_text(), "Full upstream fixture license\n"))
                converter.candidate_metadata(second, new_pack)
            for name in ("model.json", "LICENSE.txt"):
                self.assertEqual((pack / name).read_bytes(), (new_pack / name).read_bytes())
            self.assertEqual((new_pack / "LICENSE.txt").read_text().count("Conversion notice:"), 1)

    def test_numerical_gate_rejects_nan_shape_error_and_mask_divergence(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy required for numeric gate checks")
        expected = np.zeros((1, 1, 256, 256), dtype=np.float32)
        self.assertEqual(converter.compare_alpha("equal", expected, expected)["mask_iou_at_0_5"], 1)
        for actual in (np.full_like(expected, np.nan), expected[0], np.full_like(expected, 0.01)):
            with self.subTest(shape=actual.shape):
                with self.assertRaises(ValueError):
                    converter.compare_alpha("invalid", actual, expected)
        # Small absolute differences can still change the segmentation mask.
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            converter.compare_alpha("mask", np.full_like(expected, 0.500001), np.full_like(expected, 0.499999))

    def test_modified_model_rejected_before_tool_execution_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            model.write_bytes(b"wrong graph")
            with patch.object(converter.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "input integrity"):
                    converter.convert(model, root / "tool", root / "license", root / "helper", root / "output")
            run.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_symlink_rejected_even_when_its_bytes_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_bytes(b"fixture")
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "input integrity"):
                converter.verify_input(link, converter.sha256(source))

    def test_rename_only_blob_tokens_preserving_layer_names_and_parameters(self):
        graph = GRAPH.replace("ReLU relu", "ReLU in0").replace("hidden out0", "hidden out0 0=1")
        actual = converter.normalize_param(graph)
        self.assertIn("Input in0 0 1 input", actual)
        self.assertIn("ReLU in0 1 1 input hidden", actual)
        self.assertIn("hidden output 0=1", actual)

    def test_unknown_operator_incomplete_graph_counts_and_io_fail_closed(self):
        for graph in (GRAPH.replace("ReLU", "CustomCPUOp"),
                      GRAPH.replace("in0 hidden", "missing hidden"),
                      GRAPH.replace("3 3", "4 3"),
                      GRAPH.replace("3 3", "3 4"),
                      GRAPH.replace("hidden out0", "hidden out0 8=1"),
                      GRAPH.replace("hidden out0", "hidden out0 19=1"),
                      GRAPH.replace("hidden out0", "hidden another"),
                      GRAPH.replace("1 1 hidden out0", "2 1 hidden out0")):
            with self.subTest(graph=graph):
                with self.assertRaises(ValueError):
                    converter.normalize_param(graph)

    def test_candidate_metadata_and_installer_use_only_candidate_artifacts(self):
        catalog_path = ROOT / "installer/models/curated-model-catalog-v1.json"
        before = catalog_path.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pack = output / "source" / converter.PACK_PATH
            pack.mkdir(parents=True)
            for name in ("model.onnx", "model.ncnn.param", "model.ncnn.bin", "LICENSE.txt"):
                (pack / name).write_bytes((name + " fixture").encode())
            manifest = converter.candidate_metadata(output, pack)
            self.assertEqual(manifest["id"], converter.PACK_ID)
            self.assertEqual(manifest["output"]["width"], 256)
            self.assertEqual(manifest["ncnn_vulkan"]["output_blob"], manifest["output"]["name"])
            self.assertEqual(len(manifest["files"]), 3)
            for entry in manifest["files"]:
                self.assertEqual(entry["sha256"], converter.sha256(pack / entry["name"]))
            tx_spec = importlib.util.spec_from_file_location("modnet_tx", catalog_path.parent / "model_transactions.py")
            tx = importlib.util.module_from_spec(tx_spec)
            sys.modules[tx_spec.name] = tx
            tx_spec.loader.exec_module(tx)
            catalog = tx.load_catalog(output / "curated-model-catalog-v1.candidate.json")
            installer = tx.ModelTransactionInstaller(catalog, output / "metadata", output / "installed", output / "cache",
                                                     tx.FileTransport(output / "source"), retries=0)
            result = installer.install("open_video", [converter.PACK_ID])
            self.assertEqual(result["state"], "succeeded", result)
            offline = tx.ModelTransactionInstaller(catalog, output / "metadata", output / "offline", output / "cache",
                                                   tx.FileTransport(output / "unavailable"), retries=0, offline=True)
            result = offline.install("open_video", [converter.PACK_ID])
            self.assertEqual(result["state"], "succeeded", result)
            installed = output / "installed/matting/Good Quality"
            for name in ("model.onnx", "model.ncnn.param", "model.ncnn.bin", "model.json", "LICENSE.txt"):
                self.assertEqual((installed / name).read_bytes(), (pack / name).read_bytes())
            candidate = json.loads((output / "curated-model-catalog-v1.candidate.json").read_text())
            original = json.loads(before)
            for new, old in zip(candidate["packs"], original["packs"]):
                if old["id"] != converter.PACK_ID:
                    self.assertEqual(new, old)
                self.assertEqual(new["default"], old["default"])
        self.assertEqual(catalog_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
