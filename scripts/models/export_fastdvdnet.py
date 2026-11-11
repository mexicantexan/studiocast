#!/usr/bin/env python3
"""Offline, pinned FastDVDnet export; never installs or publishes a model.

Fetch the three pinned upstream files manually into --upstream. Use the isolated
tool versions documented in docs/INSTALLER_MODEL_CATALOG.md. --output must not
exist; failed exports stay isolated and are never activated by the installer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

UPSTREAM_COMMIT = "c8fdf6182a0340e89dd18f5df25b47337cbede6f"
UPSTREAM_BASE = f"https://raw.githubusercontent.com/m-tassano/fastdvdnet/{UPSTREAM_COMMIT}"
UPSTREAM_HASHES = {
    "models.py": "3b38ee3892420d1bd8b0c5b9b7d09bf44931276465c66527eff7555f2fa17532",
    "model.pth": "9d9d8413c33e3d9d961d07c530237befa1197610b9d60602ff42fd77975d2a17",
    "LICENSE": "a74a120e0719a221c65435a670305ff8a7cc58a0064dc9cdb17125363f31cece",
}
TOOL_VERSIONS = {"torch": "2.7.1+cpu", "onnx": "1.16.0", "numpy": "2.1.3", "onnxruntime": "1.23.2"}
PACK_PATH = "open_video/video_denoise/Good Quality (FastDVDnet sigma15)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_license(template: str, upstream_license: str) -> str:
    # Promotion puts the full notice in the template; reproduce those same bytes.
    marker = "\nFull upstream license at pinned commit:\n\n"
    summary = template.split(marker, 1)[0]
    return summary + marker + upstream_license


def verify_upstream(upstream: Path) -> None:
    # Verify before importing Python or deserializing even a weights-only file.
    for name, expected in UPSTREAM_HASHES.items():
        path = upstream / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"upstream integrity failure: {name}")


def validate_graph(model) -> None:
    import onnx

    if any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL
           for t in model.graph.initializer):
        raise ValueError("export must contain all weights, without external data")
    onnx.checker.check_model(model, full_check=True)
    signature = [(x.name, x.type.tensor_type.elem_type,
                  [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim])
                 for x in [*model.graph.input, *model.graph.output]]
    expected = [("noisy", 1, ["N", 15, "H", "W"]),
                ("noise_map", 1, ["N", 1, "H", "W"]),
                ("denoised", 1, ["N", 3, "H", "W"])]
    if signature != expected:
        raise ValueError(f"unexpected FP32 NCHW signature: {signature}")


def export(upstream: Path, output: Path) -> dict:
    verify_upstream(upstream)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    versions = dict(torch=torch.__version__, onnx=onnx.__version__,
                    numpy=np.__version__, onnxruntime=ort.__version__)
    if versions != TOOL_VERSIONS:
        raise ValueError(f"use pinned exporter versions {TOOL_VERSIONS}; got {versions}")
    output.mkdir(parents=True, exist_ok=False)
    pack = output / "source" / PACK_PATH
    pack.mkdir(parents=True)
    artifact = pack / "model.onnx"
    spec = importlib.util.spec_from_file_location("verified_fastdvdnet", upstream / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.manual_seed(20261110)
    network = module.FastDVDnet(num_input_frames=5)
    state = torch.load(upstream / "model.pth", map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    network.load_state_dict(state, strict=True)
    network.eval()
    with torch.inference_mode():
        noisy = torch.rand(1, 15, 16, 16)
        noise_map = torch.full((1, 1, 16, 16), 15.0 / 255.0)
        torch.onnx.export(
            network, (noisy, noise_map), artifact,
            input_names=["noisy", "noise_map"], output_names=["denoised"],
            opset_version=17, dynamo=False, export_params=True,
            do_constant_folding=True, keep_initializers_as_inputs=False,
            dynamic_axes={name: {0: "N", 2: "H", 3: "W"}
                          for name in ("noisy", "noise_map", "denoised")},
        )
    validate_graph(onnx.load(artifact, load_external_data=False))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(artifact), options, providers=["CPUExecutionProvider"])
    rows = []
    with torch.inference_mode():
        for n, h, w in ((1, 8, 8), (1, 12, 20), (2, 32, 24)):
            for sigma in (5, 15, 55):
                noisy = torch.rand(n, 15, h, w)
                noise_map = torch.full((n, 1, h, w), sigma / 255.0)
                expected = network(noisy, noise_map).numpy()
                actual, = session.run(["denoised"],
                                      {"noisy": noisy.numpy(), "noise_map": noise_map.numpy()})
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                rows.append(dict(shape=[n, 15, h, w], sigma=sigma,
                                 max_abs_error=float(np.max(np.abs(actual - expected)))))
    repo = Path(__file__).resolve().parents[2]
    template = repo / "resources/model_packs" / PACK_PATH
    manifest = json.loads((template / "model.json").read_text())
    manifest["files"][0]["sha256"] = sha256(artifact)
    manifest["origin"].update(upstream_commit=UPSTREAM_COMMIT,
                              weights_sha256=UPSTREAM_HASHES["model.pth"],
                              notes="Self-contained opset-17 FP32 export of verified upstream model.pth; "
                                    "scripts/models/export_fastdvdnet.py. Dynamic N/H/W; H/W multiples of four.")
    (pack / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (pack / "LICENSE.txt").write_text(compose_license(
        (template / "LICENSE.txt").read_text(), (upstream / "LICENSE").read_text()))
    catalog = json.loads((repo / "installer/models/curated-model-catalog-v1.json").read_text())
    catalog["catalog_version"] = "2026-11-10-fastdvdnet-candidate"
    definition = next(p for p in catalog["packs"] if p["id"] == "fastdvdnet_sigma15")
    definition["artifacts"][0].update(sha256=sha256(artifact), size_bytes=artifact.stat().st_size,
                                      source_paths=[PACK_PATH + "/model.onnx"])
    definition["provenance"]["release"] = f"upstream {UPSTREAM_COMMIT}; self-contained opset17; candidate, unpublished"
    for item in definition["metadata"]:
        item["sha256"] = sha256(pack / item["destination"])
        metadata_path = output / "metadata" / item["source"]
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pack / item["destination"], metadata_path)
    (output / "curated-model-catalog-v1.candidate.json").write_text(json.dumps(catalog, indent=2) + "\n")
    report = dict(status="candidate_unpublished", artifact=str(artifact.relative_to(output)),
                  sha256=sha256(artifact), size_bytes=artifact.stat().st_size,
                  upstream_commit=UPSTREAM_COMMIT, upstream_files=UPSTREAM_HASHES,
                  upstream_base=UPSTREAM_BASE, tools=versions, graph_check="passed_full_check",
                  metadata_root="metadata",
                  cpu_equivalence=rows, cuda_qualification="not_run",
                  promotion="requires reviewed publication and signed packaged catalog; shipped catalog unchanged")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.upstream.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
