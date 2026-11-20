#!/usr/bin/env python3

"""Generate the synthetic ONNX model used by the Open CUDA mock_model fixture.

This is a developer-only helper. The repository ships a pre-generated model at:
  tests/data/models/open_cuda/mock_model/model.onnx

Graph:
  input (float32 [1,3,256,256]) -> ReduceMean(axes=[1], keepdims=1) -> alpha (float32 [1,1,256,256])
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_model():
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 256, 256])
    out = helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1, 1, 256, 256])

    node = helper.make_node(
        "ReduceMean",
        inputs=["input"],
        outputs=["alpha"],
        axes=[1],
        keepdims=1,
    )

    graph = helper.make_graph(
        [node],
        "mock_matting",
        [inp],
        [out],
    )

    # Keep this fixture compatible with the oldest ONNX Runtime used by CI.
    # ORT 1.17.x supports ONNX IR up to 9, and opset 11 is enough for this graph.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 9

    onnx.checker.check_model(model)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tests/data/models/open_cuda/mock_model/model.onnx"),
        help="Output .onnx path (default: tests/data/models/open_cuda/mock_model/model.onnx)",
    )
    args = ap.parse_args()

    try:
        import onnx  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "onnx python package is required to generate the mock model. "
            "Install it in your dev environment (e.g. `python3 -m pip install --user onnx`).\n"
            f"Import error: {e}"
        )

    model = build_model()

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import onnx

    onnx.save(model, str(out_path))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
