#!/usr/bin/env python3
"""Hermetic exporter regression checks; no weights, downloads, or host mutation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/models/export_fastdvdnet.py"
spec = importlib.util.spec_from_file_location("fastdvdnet_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTrustTests(unittest.TestCase):
    def test_exported_candidate_installs_from_its_own_source_tree(self):
        # Mock inference only; export layout, metadata, catalog, and install are real.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            upstream.mkdir()
            (upstream / "models.py").write_text(
                "from unittest.mock import MagicMock\nFastDVDnet = MagicMock\n")
            (upstream / "model.pth").write_bytes(b"tiny checkpoint fixture")
            (upstream / "LICENSE").write_text("fixture license\n")
            hashes = {p.name: exporter.sha256(p) for p in upstream.iterdir()}
            dependencies = {name: MagicMock(__version__=version)
                            for name, version in exporter.TOOL_VERSIONS.items()}
            dependencies["torch"].load.return_value = {}
            dependencies["numpy"].testing.assert_allclose = MagicMock()
            dependencies["onnxruntime"].InferenceSession.return_value.run.return_value = [MagicMock()]
            dependencies["torch"].onnx.export.side_effect = (
                lambda network, inputs, path, **kwargs: path.write_bytes(b"tiny ONNX fixture"))
            active_catalog = SCRIPT.parents[2] / "installer/models/curated-model-catalog-v1.json"
            before = active_catalog.read_bytes()
            with patch.dict(sys.modules, dependencies), \
                    patch.dict(exporter.UPSTREAM_HASHES, hashes, clear=True), \
                    patch.object(exporter, "validate_graph"):
                output = root / "candidate"
                exporter.export(upstream, output)

            tx_spec = importlib.util.spec_from_file_location(
                "fastdvdnet_candidate_tx", active_catalog.parent / "model_transactions.py")
            tx = importlib.util.module_from_spec(tx_spec)
            sys.modules[tx_spec.name] = tx
            tx_spec.loader.exec_module(tx)
            catalog = tx.load_catalog(output / "curated-model-catalog-v1.candidate.json")
            result = tx.ModelTransactionInstaller(
                catalog, output / "metadata", root / "models", root / "cache",
                tx.FileTransport(output / "source"), retries=0,
            ).install("open_video", ["fastdvdnet_sigma15"])
            self.assertEqual(result["state"], "succeeded", result)
            pack = root / "models/video_denoise/Good Quality (FastDVDnet sigma15)"
            self.assertEqual((pack / "model.onnx").read_bytes(), b"tiny ONNX fixture")
            self.assertEqual(active_catalog.read_bytes(), before)
            active = next(p for p in json.loads(before)["packs"] if p["id"] == "fastdvdnet_sigma15")
            self.assertIn("/revisions/", active["artifacts"][0]["source_paths"][0])

    def test_reexport_preserves_exact_license_after_template_promotion(self):
        summary = "StudioCast model summary\n"
        upstream_license = "Copyright fixture\nPermission is hereby granted...\n"
        published = summary + "\nFull upstream license at pinned commit:\n\n" + upstream_license
        self.assertEqual(exporter.compose_license(summary, upstream_license), published)
        self.assertEqual(exporter.compose_license(published, upstream_license), published)

    def test_missing_and_modified_upstream_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as root:
            upstream = Path(root)
            with self.assertRaisesRegex(ValueError, "models.py"):
                exporter.verify_upstream(upstream)
            (upstream / "models.py").write_text("raise RuntimeError('must never execute')")
            with self.assertRaisesRegex(ValueError, "models.py"):
                exporter.export(upstream, upstream / "output")
            self.assertFalse((upstream / "output").exists())

    def test_upstream_symlink_rejected_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as root:
            upstream = Path(root)
            source = upstream / "source.py"
            source.write_text("verified fixture")
            (upstream / "models.py").symlink_to(source)
            with patch.dict(exporter.UPSTREAM_HASHES,
                            {"models.py": exporter.sha256(source)}, clear=True):
                with self.assertRaisesRegex(ValueError, "models.py"):
                    exporter.verify_upstream(upstream)


class GraphRegressionTests(unittest.TestCase):
    def setUp(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("ONNX required for exporter graph checks")
        self.onnx = onnx

    def graph(self, missing_weight=False):
        h, t = self.onnx.helper, self.onnx.TensorProto
        inputs = [h.make_tensor_value_info("noisy", t.FLOAT, ["N", 15, "H", "W"]),
                  h.make_tensor_value_info("noise_map", t.FLOAT, ["N", 1, "H", "W"])]
        output = h.make_tensor_value_info("denoised", t.FLOAT, ["N", 3, "H", "W"])
        # Slice the center RGB frame, as a small valid graph with the exact signature.
        initializers = [h.make_tensor(name, t.INT64, [1], [value])
                        for name, value in (("starts", 6), ("ends", 9), ("axes", 1))]
        node = h.make_node("Slice", ["noisy", "absent" if missing_weight else "starts", "ends", "axes"], ["denoised"])
        return h.make_model(h.make_graph([node], "fixture", inputs, [output], initializers),
                            opset_imports=[h.make_opsetid("", 17)])

    def test_self_contained_valid_graph_passes(self):
        exporter.validate_graph(self.graph())

    def test_missing_initializer_is_rejected(self):
        with self.assertRaises(self.onnx.checker.ValidationError):
            exporter.validate_graph(self.graph(missing_weight=True))

    def test_external_weight_reference_is_rejected(self):
        graph = self.graph()
        tensor = graph.graph.initializer[0]
        tensor.data_location = self.onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = "location", "fastdvdnet.onnx.data"
        with self.assertRaisesRegex(ValueError, "without external data"):
            exporter.validate_graph(graph)


if __name__ == "__main__":
    unittest.main()
