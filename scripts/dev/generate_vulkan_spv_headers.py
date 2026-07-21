#!/usr/bin/env python3
"""Compile Vulkan GLSL compute shaders and embed SPIR-V headers."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from pathlib import Path
import subprocess
import tempfile


@dataclasses.dataclass(frozen=True)
class ShaderHeader:
    source: str
    header: str
    array_name: str
    size_name: str
    hash_name: str


SHADERS = (
    ShaderHeader(
        source="src/core/vulkan/kernels/shaders/resize_rgb24_bilinear.comp",
        header="src/core/vulkan/kernels/shaders/resize_rgb24_bilinear_spv.h",
        array_name="kResizeRgb24BilinearSpirv",
        size_name="kResizeRgb24BilinearSpirvSizeBytes",
        hash_name="kResizeRgb24BilinearShaderSourceSha256",
    ),
    ShaderHeader(
        source="src/core/vulkan/kernels/shaders/utility_kernels.comp",
        header="src/core/vulkan/kernels/shaders/utility_kernels_spv.h",
        array_name="kUtilityKernelsSpirv",
        size_name="kUtilityKernelsSpirvSizeBytes",
        hash_name="kUtilityKernelsShaderSourceSha256",
    ),
)


def compile_shader(glslang: Path, root: Path, shader: ShaderHeader) -> bytes:
    src = root / shader.source
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
                str(src),
            ],
            cwd=root,
            check=True,
        )
        return out.read_bytes()


def format_words(spv: bytes) -> str:
    if len(spv) % 4 != 0:
        raise ValueError("SPIR-V bytecode length is not a multiple of 4")
    words = [
        int.from_bytes(spv[i : i + 4], byteorder="little")
        for i in range(0, len(spv), 4)
    ]
    lines: list[str] = []
    for i in range(0, len(words), 6):
        chunk = words[i : i + 6]
        lines.append(
            "    " + ", ".join(f"0x{word:08x}" for word in chunk) + ","
        )
    return "\n".join(lines)


def write_header(root: Path, shader: ShaderHeader, spv: bytes) -> None:
    source_bytes = (root / shader.source).read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    text = f"""#pragma once

#include <cstddef>
#include <cstdint>

namespace studiocast::vulkan::kernels::shaders {{

inline constexpr const char *{shader.hash_name} =
    "{source_hash}";

inline constexpr std::uint32_t {shader.array_name}[] = {{
{format_words(spv)}
}};

inline constexpr std::size_t {shader.size_name} =
    sizeof({shader.array_name});

}} // namespace studiocast::vulkan::kernels::shaders
"""
    (root / shader.header).write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--glslang-validator", required=True)
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    glslang = Path(args.glslang_validator).resolve()
    for shader in SHADERS:
        spv = compile_shader(glslang, root, shader)
        write_header(root, shader, spv)
        print(f"generated {shader.header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
