#!/usr/bin/env python3
"""Validate embedded PTX provenance and optional nvcc freshness.

The no-nvcc path is intentionally useful in normal CI: every embedded PTX block
must be listed here, have a .cu source, and document its generation command at
the embedding site. When nvcc is available, the script also compiles those .cu
sources and verifies the generated PTX exposes the same entry ABI.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


@dataclasses.dataclass(frozen=True)
class PtxModule:
    name: str
    ptx_cpp: str
    ptx_symbol: str
    cu_source: str
    command: str
    nvcc_args: tuple[str, ...]
    entries: tuple[str, ...]
    exact_toolchain: str | None = None
    required_embedded_fragments: tuple[str, ...] = ()
    forbidden_embedded_fragments: tuple[str, ...] = ()
    required_source_fragments: tuple[str, ...] = ()
    forbidden_source_fragments: tuple[str, ...] = ()


OPEN_CUDA_NVCC_ARGS = (
    "-ptx",
    "-O3",
    "--use_fast_math",
    "-arch=compute_52",
    "-I",
    "src",
)

CUDA_NVCC_ARGS = (
    "-ptx",
    "-O3",
    "-arch=compute_52",
    "-I",
    "src",
)

MODULES = (
    PtxModule(
        name="open_cuda_vb",
        ptx_cpp="src/core/cuda/kernels/open_cuda_vb_kernels_ptx.cpp",
        ptx_symbol="kOpenCudaVbPtx",
        cu_source="src/core/cuda/kernels/open_cuda_vb_kernels.cu",
        command=(
            "nvcc -ptx -O3 --use_fast_math -arch=compute_52 -I src "
            "src/core/cuda/kernels/open_cuda_vb_kernels.cu "
            "-o open_cuda_vb_kernels.ptx"
        ),
        nvcc_args=OPEN_CUDA_NVCC_ARGS,
        entries=(
            "resize_bilinear_f32_1",
            "box_blur_h_u8x3",
            "box_blur_v_u8x3",
            "box_blur_h_f32_1",
            "box_blur_v_f32_1",
            "composite_alpha_u8x3_bg",
            "composite_alpha_u8x3_solid",
            "key_light_u8x3",
        ),
        exact_toolchain="Cuda compilation tools, release 12.6, V12.6.85",
    ),
    PtxModule(
        name="resize_bilinear",
        ptx_cpp="src/core/cuda/kernels/resize_bilinear_ptx.cpp",
        ptx_symbol="kResizeBilinearPtx",
        cu_source="src/core/cuda/kernels/resize_bilinear.cu",
        command=(
            "nvcc -ptx -O3 -arch=compute_52 -I src "
            "src/core/cuda/kernels/resize_bilinear.cu "
            "-o resize_bilinear.ptx"
        ),
        nvcc_args=CUDA_NVCC_ARGS,
        entries=("resize_bilinear_u8x3",),
        required_embedded_fragments=("cvt.rzi.u32.f32",),
        forbidden_embedded_fragments=("cvt.rni.u32.f32",),
    ),
    PtxModule(
        name="preprocess_to_nchw",
        ptx_cpp="src/core/cuda/kernels/preprocess_to_nchw_ptx.cpp",
        ptx_symbol="kPreprocessPtx",
        cu_source="src/core/cuda/kernels/preprocess_to_nchw.cu",
        command=(
            "nvcc -ptx -O3 -arch=compute_52 -I src "
            "src/core/cuda/kernels/preprocess_to_nchw.cu "
            "-o preprocess_to_nchw.ptx"
        ),
        nvcc_args=CUDA_NVCC_ARGS,
        entries=("preprocess_to_nchw_f32",),
    ),
    PtxModule(
        name="maxine_crop_scale",
        ptx_cpp="src/core/maxine/cuda_crop_scale.cpp",
        ptx_symbol="kCropScalePtx",
        cu_source="src/core/maxine/cuda_crop_scale_kernels.cu",
        command=(
            "nvcc -ptx -O3 -arch=compute_52 -I src "
            "src/core/maxine/cuda_crop_scale_kernels.cu "
            "-o cuda_crop_scale_kernels.ptx"
        ),
        nvcc_args=CUDA_NVCC_ARGS,
        entries=("crop_scale_bgr_u8",),
    ),
    PtxModule(
        name="maxine_resize_bilinear",
        ptx_cpp="src/core/maxine/cuda_crop_scale.cpp",
        ptx_symbol="kResizeBilinearPtx",
        cu_source="src/core/maxine/cuda_crop_scale_kernels.cu",
        command=(
            "nvcc -ptx -O3 -arch=compute_52 -I src "
            "src/core/maxine/cuda_crop_scale_kernels.cu "
            "-o cuda_crop_scale_kernels.ptx"
        ),
        nvcc_args=CUDA_NVCC_ARGS,
        entries=("resize_bilinear_bgr_u8",),
        required_embedded_fragments=("cvt.rzi.u32.f32",),
        forbidden_embedded_fragments=("cvt.rni.u32.f32",),
        required_source_fragments=("RoundClampHalfUpToU8(v)",),
        forbidden_source_fragments=("RoundNearestToU8(v + 0.5f)",),
    ),
    PtxModule(
        name="maxine_vignette",
        ptx_cpp="src/core/maxine/cuda_vignette.cpp",
        ptx_symbol="kVignettePtx",
        cu_source="src/core/maxine/cuda_vignette_kernel.cu",
        command=(
            "nvcc -ptx -O3 -arch=compute_52 -I src "
            "src/core/maxine/cuda_vignette_kernel.cu "
            "-o cuda_vignette_kernel.ptx"
        ),
        nvcc_args=CUDA_NVCC_ARGS,
        entries=("vignette_bgr_u8",),
    ),
)

PTX_BLOCK_RE = re.compile(
    r"static\s+constexpr\s+const\s+char\s*\*\s*(?P<symbol>\w+)\s*=\s*"
    r"R\"ptx\(\n(?P<body>.*?)\n\)ptx\";",
    re.DOTALL,
)
ENTRY_RE = re.compile(
    r"\.visible\s+\.entry\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\((?P<params>.*?)\)\s*\{",
    re.DOTALL,
)
PARAM_RE = re.compile(r"\.param\s+(?P<type>\.[A-Za-z0-9_]+)\s+\w+")
TOOLCHAIN_RE = re.compile(r"Cuda compilation tools, release [^\n]+")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_ptx(ptx: str) -> str:
    lines = ptx.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines).strip() + "\n"


def parse_entries(ptx: str) -> dict[str, tuple[str, ...]]:
    entries: dict[str, tuple[str, ...]] = {}
    for match in ENTRY_RE.finditer(ptx):
        params = tuple(p.group("type") for p in PARAM_RE.finditer(match.group("params")))
        entries[match.group("name")] = params
    return entries


def toolchain_id(ptx: str) -> str | None:
    match = TOOLCHAIN_RE.search(ptx)
    return match.group(0) if match else None


def discover_ptx_blocks(root: Path) -> dict[tuple[str, str], str]:
    blocks: dict[tuple[str, str], str] = {}
    for path in (root / "src").rglob("*.cpp"):
        rel = path.relative_to(root).as_posix()
        text = read_text(path)
        for match in PTX_BLOCK_RE.finditer(text):
            blocks[(rel, match.group("symbol"))] = match.group("body")
    return blocks


def validate_provenance(root: Path) -> tuple[int, dict[tuple[str, str], str]]:
    failures = 0
    blocks = discover_ptx_blocks(root)
    manifest_keys = {(m.ptx_cpp, m.ptx_symbol) for m in MODULES}
    discovered_keys = set(blocks)

    for key in sorted(discovered_keys - manifest_keys):
        print(f"[FAIL] embedded PTX block is not in validator manifest: {key[0]}:{key[1]}")
        failures += 1
    for key in sorted(manifest_keys - discovered_keys):
        print(f"[FAIL] manifest PTX block was not found: {key[0]}:{key[1]}")
        failures += 1

    for module in MODULES:
        ptx_path = root / module.ptx_cpp
        cu_path = root / module.cu_source
        if not cu_path.is_file():
            print(f"[FAIL] {module.name}: missing .cu source {module.cu_source}")
            failures += 1
        else:
            cu_text = read_text(cu_path)
            for fragment in module.required_source_fragments:
                if fragment not in cu_text:
                    print(
                        f"[FAIL] {module.name}: .cu source is missing required fragment "
                        f"{fragment!r}"
                    )
                    failures += 1
            for fragment in module.forbidden_source_fragments:
                if fragment in cu_text:
                    print(
                        f"[FAIL] {module.name}: .cu source contains forbidden fragment "
                        f"{fragment!r}"
                    )
                    failures += 1
        if not ptx_path.is_file():
            continue

        text = read_text(ptx_path)
        source_tag = f"PTX_SOURCE: {module.cu_source}"
        if source_tag not in text:
            print(f"[FAIL] {module.name}: missing '{source_tag}' in {module.ptx_cpp}")
            failures += 1
        if module.command not in text:
            print(f"[FAIL] {module.name}: missing documented command in {module.ptx_cpp}")
            failures += 1

        body = blocks.get((module.ptx_cpp, module.ptx_symbol))
        if body is None:
            continue
        embedded_entries = parse_entries(body)
        if set(embedded_entries) != set(module.entries):
            found = ", ".join(sorted(embedded_entries)) or "(none)"
            expected = ", ".join(module.entries)
            print(f"[FAIL] {module.name}: embedded entries {found}; expected {expected}")
            failures += 1
        for fragment in module.required_embedded_fragments:
            if fragment not in body:
                print(
                    f"[FAIL] {module.name}: embedded PTX is missing required fragment "
                    f"{fragment!r}"
                )
                failures += 1
        for fragment in module.forbidden_embedded_fragments:
            if fragment in body:
                print(
                    f"[FAIL] {module.name}: embedded PTX contains forbidden fragment "
                    f"{fragment!r}"
                )
                failures += 1

    if failures == 0:
        print("[OK] embedded PTX provenance metadata is complete")
    return failures, blocks


def nvcc_path() -> str | None:
    from_env = os.environ.get("STUDIOCAST_NVCC")
    if from_env:
        return from_env
    return shutil.which("nvcc")


def compile_sources(root: Path, nvcc: str, tmp: Path) -> tuple[int, dict[tuple[str, tuple[str, ...]], str]]:
    failures = 0
    generated: dict[tuple[str, tuple[str, ...]], str] = {}
    unique_sources = sorted({(m.cu_source, m.nvcc_args) for m in MODULES})
    for source, args in unique_sources:
        out_name = source.replace("/", "_").replace(".", "_") + ".ptx"
        out_path = tmp / out_name
        cmd = [nvcc, *args, source, "-o", str(out_path)]
        result = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"[FAIL] nvcc failed for {source}")
            if result.stdout:
                print(result.stdout.rstrip())
            if result.stderr:
                print(result.stderr.rstrip())
            failures += 1
            continue
        generated[(source, args)] = read_text(out_path)
    return failures, generated


def validate_nvcc(root: Path, blocks: dict[tuple[str, str], str], require_exact: bool) -> int:
    nvcc = nvcc_path()
    if not nvcc:
        print("[SKIP] nvcc not found; PTX compile comparison skipped")
        return 0

    failures = 0
    with tempfile.TemporaryDirectory(prefix="studiocast-ptx-") as tmp_dir:
        compile_failures, generated = compile_sources(root, nvcc, Path(tmp_dir))
        failures += compile_failures
        if compile_failures:
            return failures

        for module in MODULES:
            embedded = blocks.get((module.ptx_cpp, module.ptx_symbol))
            generated_ptx = generated.get((module.cu_source, module.nvcc_args))
            if embedded is None or generated_ptx is None:
                continue

            embedded_entries = parse_entries(embedded)
            generated_entries = parse_entries(generated_ptx)
            for entry in module.entries:
                if entry not in generated_entries:
                    print(f"[FAIL] {module.name}: generated PTX is missing entry {entry}")
                    failures += 1
                    continue
                if generated_entries[entry] != embedded_entries.get(entry):
                    print(
                        f"[FAIL] {module.name}: entry ABI mismatch for {entry}: "
                        f"embedded {embedded_entries.get(entry)} generated {generated_entries[entry]}"
                    )
                    failures += 1

            if module.exact_toolchain:
                generated_toolchain = toolchain_id(generated_ptx)
                embedded_toolchain = toolchain_id(embedded)
                if (
                    generated_toolchain == module.exact_toolchain
                    and embedded_toolchain == module.exact_toolchain
                ):
                    if normalize_ptx(generated_ptx) != normalize_ptx(embedded):
                        print(f"[FAIL] {module.name}: generated PTX differs from embedded PTX")
                        failures += 1
                    else:
                        print(f"[OK] {module.name}: exact PTX matches {module.exact_toolchain}")
                else:
                    message = (
                        f"{module.name}: exact PTX comparison needs "
                        f"{module.exact_toolchain}; generated {generated_toolchain or 'unknown'}"
                    )
                    if require_exact:
                        print(f"[FAIL] {message}")
                        failures += 1
                    else:
                        print(f"[SKIP] {message}")

    if failures == 0:
        print("[OK] nvcc PTX entry ABI comparison passed")
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=Path(__file__).resolve().parents[3],
        type=Path,
        help="Repository source root",
    )
    parser.add_argument(
        "--require-nvcc",
        action="store_true",
        help="Fail when nvcc is unavailable",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail when exact generated PTX comparison is skipped",
    )
    args = parser.parse_args(argv)

    root = args.source_root.resolve()
    failures, blocks = validate_provenance(root)

    require_nvcc = args.require_nvcc or os.environ.get(
        "STUDIOCAST_REQUIRE_NVCC_PTX_FRESHNESS", ""
    ).lower() in {"1", "true", "yes", "on"}
    if require_nvcc and not nvcc_path():
        print("[FAIL] nvcc is required but was not found")
        failures += 1
    else:
        failures += validate_nvcc(root, blocks, args.require_exact)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
