#!/usr/bin/env python3
"""Offline conversion of the exact curated MODNet; never installs or publishes.

Inputs are existing ONNX, extracted pinned pnnx executable, full upstream license,
and the compiled modnet_ncnn_reference.cpp helper. See modnet-ncnn.lock.json for
immutable tooling/source identities. --output must not exist. Candidate catalogs
point to their own source tree, leaving shipped metadata and installed packs alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads(Path(__file__).with_name("modnet-ncnn.lock.json").read_text())
PACK_ID = "modnet-webnn-256-fp32"
PACK_PATH = "open_video/matting/Good Quality"
# Reviewed implementations at ncnn_commit; runtime must still reject any layer
# lacking Vulkan support after load. Conversion alone does not prove residency.
VULKAN_TYPES = frozenset(("Input", "Split", "Convolution", "ConvolutionDepthWise",
                         "BinaryOp", "BatchNorm", "InstanceNorm", "Slice",
                         "Concat", "ReLU", "Interp", "InnerProduct", "Pooling",
                         "Reshape"))


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def verify_input(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"input integrity failure: {path.name}")


def compose_license(template: str, upstream: str) -> str:
    # A promoted template already contains both generated sections. Strip the
    # first generated marker so the same approved bytes reproduce afterward.
    for marker in ("\nConversion notice:", "\nFull upstream license"):
        template = template.split(marker, 1)[0]
    return (template +
        "\nConversion notice: StudioCast converted the original ONNX to fixed-256 FP32 ncnn\n"
        "with pnnx 20261020 and restored external blob names input/output; no retraining.\n"
        "Full upstream license at MODNet commit 28165a451e4610c9d77cfdf925a94610bb2810fb:\n\n" + upstream)


def normalize_param(text: str) -> str:
    """Rename blob tokens, preserving ONNX input/output and layer parameters."""
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "7767517":
        raise ValueError("invalid ncnn param header")
    declared_layers, declared_blobs = map(int, lines[1].split())
    rows, produced, consumed = [], set(), set()
    for line in lines[2:]:
        tokens = line.split()
        if len(tokens) < 4 or tokens[0] not in VULKAN_TYPES:
            raise ValueError("unreviewed ncnn layer type")
        bottom_count, top_count = map(int, tokens[2:4])
        if min(bottom_count, top_count) < 0 or len(tokens) < 4 + bottom_count + top_count:
            raise ValueError("invalid ncnn blob counts")
        bottom = tokens[4:4 + bottom_count]
        top = tokens[4 + bottom_count:4 + bottom_count + top_count]
        if len(set(top)) != len(top) or any(name not in produced for name in bottom) or any(name in produced for name in top):
            raise ValueError("ncnn graph is not topologically complete")
        consumed.update(bottom)
        produced.update(top)
        # pnnx always rewrites the external names. Do not rename layer names or
        # parameter values, and reject an unexpected changed converter contract.
        end = 4 + bottom_count + top_count
        if tokens[0] in {"Convolution", "ConvolutionDepthWise", "InnerProduct"}:
            parameters = dict(token.split("=", 1) for token in tokens[end:])
            if parameters.get("8", "0") != "0" or parameters.get("19", "0") != "0":
                raise ValueError("FP32 Vulkan conversion rejects quantized/dynamic weights")
        tokens[4:end] = [{"in0": "input", "out0": "output"}.get(x, x)
                         for x in tokens[4:end]]
        rows.append(" ".join(tokens))
    inputs = [r.split()[4:] for r in rows if r.split()[0] == "Input"]
    if inputs != [["input"]] or produced - consumed != {"out0"}:
        raise ValueError("unexpected ncnn input/output signature")
    if len(rows) != declared_layers or len(produced) != declared_blobs:
        raise ValueError("ncnn graph counts disagree")
    return "\n".join([lines[0], lines[1], *rows]) + "\n"


def validate_onnx(path: Path) -> None:
    import onnx
    model = onnx.load(path, load_external_data=False)
    if any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL
           for t in model.graph.initializer):
        raise ValueError("external ONNX data is not supported")
    onnx.checker.check_model(model, full_check=True)
    signature = [(x.name, x.type.tensor_type.elem_type,
                  [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim])
                 for x in [*model.graph.input, *model.graph.output]]
    expected = [("input", 1, ["batch_size", 3, "height", "width"]),
                ("output", 1, ["batch_size", 1, "height", "width"])]
    if signature != expected:
        raise ValueError(f"unexpected original MODNet signature: {signature}")


def compare_alpha(name, actual, expected) -> dict:
    import numpy as np
    if actual.shape != (1, 1, 256, 256) or actual.shape != expected.shape:
        raise ValueError(f"incorrect alpha shape: {name}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"nonfinite alpha: {name}")
    difference = np.abs(actual - expected)
    maximum, mean = float(difference.max()), float(difference.mean())
    intersection = np.logical_and(actual >= 0.5, expected >= 0.5).sum()
    union = np.logical_or(actual >= 0.5, expected >= 0.5).sum()
    iou = float(intersection / union) if union else 1.0
    # Fixed before collecting results. Artifact equivalence never substitutes
    # for camera quality or shared-device GPU execution tests.
    if maximum > 2e-4 or mean > 2e-5 or iou < 0.999:
        raise ValueError(f"numerical mismatch {name}: max={maximum}, mae={mean}, iou={iou}")
    return dict(case=name, max_abs_error=maximum, mean_abs_error=mean,
                mask_iou_at_0_5=iou, binary_mask_disagreement_count=int(np.count_nonzero((actual >= 0.5) != (expected >= 0.5))),
                output_shape=list(actual.shape), alpha_min=float(actual.min()), alpha_max=float(actual.max()))


def numerical_validation(onnx_path: Path, pack: Path, reference: Path, work: Path) -> list:
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    axis = np.linspace(-1, 1, 256, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    cases = {"black": np.full((3, 256, 256), -1, dtype=np.float32),
             "white": np.ones((3, 256, 256), dtype=np.float32),
             "gray": np.zeros((3, 256, 256), dtype=np.float32),
             "gradient": np.stack((xx, yy, (xx + yy) / 2)),
             "checker": np.broadcast_to(((np.indices((256, 256)).sum(0) // 8) % 2) * 2 - 1,
                                         (3, 256, 256)).astype(np.float32),
             "seeded_noise": np.random.default_rng(20261111).uniform(-1, 1, (3, 256, 256)).astype(np.float32)}
    expected_cases = {name: session.run(["output"], {"input": tensor[None]})[0]
                      for name, tensor in cases.items()}
    # Preserve all golden tensors even if the first candidate comparison fails.
    for name, tensor in cases.items():
        input_path = work / (name + ".input.f32")
        tensor.astype("<f4").tofile(input_path)
        expected_cases[name].astype("<f4").tofile(work / (name + ".expected.f32"))
    (work / "fixtures.json").write_text(json.dumps(dict(dtype="float32", byte_order="little",
        input_shape=[1, 3, 256, 256], output_shape=[1, 1, 256, 256],
        cases=[dict(name=name, input=name + ".input.f32", expected=name + ".expected.f32") for name in cases]), indent=2) + "\n")
    rows = []
    for name in cases:
        expected = expected_cases[name]
        input_path = work / (name + ".input.f32")
        output = work / (name + ".ncnn.f32")
        output.unlink(missing_ok=True)
        subprocess.run([str(reference), str(pack / "model.ncnn.param"),
                        str(pack / "model.ncnn.bin"), str(input_path),
                        str(output), "input", "output"], check=True, timeout=120)
        actual = np.fromfile(output, dtype="<f4").reshape(1, 1, 256, 256)
        row = compare_alpha(name, actual, expected)
        row["fixture_files"] = dict(input=name + ".input.f32", expected=name + ".expected.f32", actual=name + ".ncnn.f32")
        rows.append(row)
    return rows


def candidate_metadata(output: Path, pack: Path) -> dict:
    template = ROOT / "resources/model_packs" / PACK_PATH
    manifest = json.loads((template / "model.json").read_text())
    manifest["output"].update(layout="nchw", width=256, height=256, channels=1)
    manifest["files"] = [f for f in manifest["files"] if f["kind"] not in ("ncnn_param", "ncnn_bin")]
    next(f for f in manifest["files"] if f["name"] == "model.onnx")["sha256"] = sha256(pack / "model.onnx")
    for name, kind in (("model.ncnn.param", "ncnn_param"), ("model.ncnn.bin", "ncnn_bin")):
        manifest["files"].append(dict(name=name, kind=kind, role="vulkan_matting", sha256=sha256(pack / name)))
    manifest["ncnn_vulkan"] = dict(param_file="model.ncnn.param", bin_file="model.ncnn.bin",
                                    input_blob="input", output_blob="output", precision="fp32",
                                    converter=dict(name="pnnx", version=LOCK["pnnx_version"]))
    manifest["origin"] = dict(onnx_sha256=LOCK["onnx_sha256"],
                               url="https://huggingface.co/onnx-community/modnet-webnn",
                               revision=LOCK["onnx_upstream_revision"],
                               notes="Original ONNX retained; offline fixed-256 FP32 ncnn conversion via scripts/models/convert_modnet_ncnn.py; no retraining.")
    (pack / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    catalog = json.loads((ROOT / "installer/models/curated-model-catalog-v1.json").read_text())
    catalog["catalog_version"] = "2026-11-11-modnet-ncnn-candidate"
    definition = next(p for p in catalog["packs"] if p["id"] == PACK_ID)
    definition["artifacts"] = [dict(name=name, sha256=sha256(pack / name),
                                     size_bytes=(pack / name).stat().st_size,
                                     size_status="known", source_paths=[PACK_PATH + "/" + name])
                               for name in ("model.onnx", "model.ncnn.param", "model.ncnn.bin")]
    definition["provenance"]["release"] = "pnnx20261020 FP32; pinned source ONNX retained; candidate unpublished"
    for item in definition["metadata"]:
        item["sha256"] = sha256(pack / item["destination"])
        destination = output / "metadata" / item["source"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pack / item["destination"], destination)
    (output / "curated-model-catalog-v1.candidate.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return manifest


def convert(onnx_path: Path, pnnx: Path, license_path: Path, reference: Path, output: Path) -> dict:
    # All trust checks precede importing heavy parsers or executing tooling.
    verify_input(onnx_path, LOCK["onnx_sha256"])
    verify_input(pnnx, LOCK["pnnx_sha256"])
    verify_input(license_path, LOCK["license_sha256"])
    if onnx_path.stat().st_size != LOCK["onnx_size_bytes"]:
        raise ValueError("incorrect source ONNX size")
    if reference.is_symlink() or not reference.is_file():
        raise ValueError("reference must be a regular executable")
    import numpy, onnx, onnxruntime
    versions = {"numpy": numpy.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__}
    if versions != LOCK["python_versions"]:
        raise ValueError(f"use pinned versions {LOCK['python_versions']}; got {versions}")
    validate_onnx(onnx_path)
    description = json.loads(subprocess.check_output([str(reference), "--describe"], text=True, timeout=10))
    if description.get("protocol") != "studiocast-modnet-ncnn-reference-v1":
        raise ValueError("unsupported native reference helper")
    output.mkdir(parents=True, exist_ok=False)
    pack = output / "source" / PACK_PATH
    pack.mkdir(parents=True)
    shutil.copyfile(onnx_path, pack / "model.onnx")
    with (output / "conversion.log").open("w") as log:
        subprocess.run([str(pnnx), "model.onnx", "inputshape=[1,3,256,256]", "fp16=0", "optlevel=2",
                        "ncnnparam=model.ncnn.param", "ncnnbin=model.ncnn.bin"], cwd=pack,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    param = pack / "model.ncnn.param"
    param.write_text(normalize_param(param.read_text()))
    verify_input(pack / "model.onnx", LOCK["onnx_sha256"])
    # Keep only the actual pack artifacts; generated Python is never executed.
    for file in pack.iterdir():
        if file.name not in {"model.onnx", "model.ncnn.param", "model.ncnn.bin"}:
            file.unlink()
    summary = (ROOT / "resources/model_packs" / PACK_PATH / "LICENSE.txt").read_text()
    (pack / "LICENSE.txt").write_text(compose_license(summary, license_path.read_text()))
    work = output / "numerical"
    work.mkdir()
    rows = numerical_validation(onnx_path, pack, reference, work)
    candidate_metadata(output, pack)
    report = dict(status="candidate_unpublished", pack_id=PACK_ID,
                  lock=LOCK, tools=versions, reference_description=description, reference_executable_sha256=sha256(reference),
                  artifacts={p.name: dict(sha256=sha256(p), size_bytes=p.stat().st_size) for p in sorted(pack.iterdir())},
                  input_contract="FP32 RGB NCHW 1x3x256x256, normalized to [-1,1]",
                  cpu_acceptance=dict(max_abs_error=2e-4, mean_abs_error=2e-5, minimum_mask_iou_at_0_5=0.999),
                  cpu_equivalence=rows, gpu_qualification="not_run",
                  promotion="requires reviewed shared-device GPU evidence and separately approved publication; shipped catalog unchanged")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "pnnx", "license", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    # Keep symlink identity available to verify_input; resolve only parents.
    paths = [p.absolute() for p in (args.onnx, args.pnnx, args.license, args.reference, args.output)]
    print(json.dumps(convert(*paths), indent=2))


if __name__ == "__main__":
    main()
