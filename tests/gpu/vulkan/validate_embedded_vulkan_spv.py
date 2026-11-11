#!/usr/bin/env python3
"""Validate embedded Vulkan SPIR-V shader provenance.

The default path is intentionally no-GPU and no-SDK safe: it checks that each
committed shader header records the current GLSL source SHA-256 and contains a
plausible SPIR-V module. If --glslang-validator is supplied, it also recompiles
and byte-compares the generated SPIR-V.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile


@dataclasses.dataclass(frozen=True)
class ShaderHeader:
    name: str
    source: str
    header: str
    array_name: str
    size_name: str
    hash_name: str


SHADERS = (
    ShaderHeader(
        name="resize_rgb24_bilinear",
        source="src/core/vulkan/kernels/shaders/resize_rgb24_bilinear.comp",
        header="src/core/vulkan/kernels/shaders/resize_rgb24_bilinear_spv.h",
        array_name="kResizeRgb24BilinearSpirv",
        size_name="kResizeRgb24BilinearSpirvSizeBytes",
        hash_name="kResizeRgb24BilinearShaderSourceSha256",
    ),
    ShaderHeader(
        name="utility_kernels",
        source="src/core/vulkan/kernels/shaders/utility_kernels.comp",
        header="src/core/vulkan/kernels/shaders/utility_kernels_spv.h",
        array_name="kUtilityKernelsSpirv",
        size_name="kUtilityKernelsSpirvSizeBytes",
        hash_name="kUtilityKernelsShaderSourceSha256",
    ),
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def embedded_words(header_text: str, array_name: str) -> list[int] | None:
    match = re.search(
        rf"{re.escape(array_name)}\[\]\s*=\s*\{{(?P<body>.*?)\}};",
        header_text,
        re.DOTALL,
    )
    if not match:
        return None
    return [int(word, 16) for word in re.findall(r"0x[0-9a-fA-F]{8}", match.group("body"))]


def compile_shader(glslang: Path, root: Path, source: str) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "shader.spv"
        subprocess.run(
            [
                str(glslang),
                "-V",
                "-S",
                "comp",
                "-o",
                str(out),
                str(root / source),
            ],
            cwd=root,
            check=True,
        )
        return out.read_bytes()


def words_to_bytes(words: list[int]) -> bytes:
    return b"".join(word.to_bytes(4, byteorder="little") for word in words)


def validate(root: Path, glslang: Path | None) -> int:
    failures = 0
    for shader in SHADERS:
        source_path = root / shader.source
        header_path = root / shader.header
        if not source_path.is_file():
            print(f"[FAIL] {shader.name}: missing source {shader.source}")
            failures += 1
            continue
        if not header_path.is_file():
            print(f"[FAIL] {shader.name}: missing header {shader.header}")
            failures += 1
            continue

        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        header_text = read_text(header_path)
        expected_hash_line = f'{shader.hash_name} =\n    "{source_hash}"'
        if expected_hash_line not in header_text:
            print(
                f"[FAIL] {shader.name}: embedded source hash is stale "
                f"(expected {source_hash})"
            )
            failures += 1

        words = embedded_words(header_text, shader.array_name)
        if not words:
            print(f"[FAIL] {shader.name}: missing embedded word array")
            failures += 1
            continue
        if words[0] != 0x07230203:
            print(f"[FAIL] {shader.name}: embedded module has bad SPIR-V magic")
            failures += 1
        if f"sizeof({shader.array_name})" not in header_text:
            print(f"[FAIL] {shader.name}: missing size constant provenance")
            failures += 1

        if glslang is not None:
            compiled = compile_shader(glslang, root, shader.source)
            embedded = words_to_bytes(words)
            if compiled != embedded:
                print(f"[FAIL] {shader.name}: embedded SPIR-V is not fresh")
                failures += 1

    if failures == 0:
        print("[PASS] embedded Vulkan SPIR-V provenance")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--glslang-validator")
    args = parser.parse_args()

    glslang = Path(args.glslang_validator).resolve() if args.glslang_validator else None
    return 1 if validate(Path(args.source_root).resolve(), glslang) else 0


if __name__ == "__main__":
    raise SystemExit(main())
