#!/usr/bin/env python3
"""Stage the fixed Microsoft ONNX Runtime 1.17.3 Linux x64 CPU SDK for CI.

Accept only the pinned official Microsoft.ML.OnnxRuntime NuGet archive. No
network, package metadata execution, or system installation is performed.
"""
import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import stat
import xml.etree.ElementTree as ET
import zipfile

# Official CPU C++ distribution: https://onnxruntime.ai/docs/get-started/with-cpp.html
# https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.17.3/microsoft.ml.onnxruntime.1.17.3.nupkg
ARCHIVE_SIZE = 83662121
ARCHIVE_SHA256 = "e7fecf6fe6ae54d5e43664c4e2354138b5b86d4f0ccf946fbab1ef48650524e3"
HEADERS = (
    "onnxruntime_cxx_inline.h", "onnxruntime_cxx_api.h", "onnxruntime_c_api.h",
    "onnxruntime_float16.h", "onnxruntime_lite_custom_op.h",
    "onnxruntime_run_options_config_keys.h",
    "onnxruntime_session_options_config_keys.h", "provider_options.h",
    "cpu_provider_factory.h",
)
MEMBERS = {"build/native/include/" + name: "include/" + name for name in HEADERS}
MEMBERS.update({
    "runtimes/linux-x64/native/libonnxruntime.so": "lib/libonnxruntime.so.1.17.3",
    "LICENSE": "LICENSE", "ThirdPartyNotices.txt": "ThirdPartyNotices.txt",
})
NUSPEC = "Microsoft.ML.OnnxRuntime.nuspec"


def prepare(archive, output):
    archive, output = Path(archive), Path(output).absolute()
    if (output.exists() or output.is_symlink() or ".." in output.parts or
            any(parent.is_symlink() for parent in output.parents)):
        raise ValueError("output must be new, without symlink parents or '..'")
    if not output.parent.is_dir():
        raise ValueError("output parent must exist")
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_nlink != 1:
        raise ValueError("archive must be a regular, non-linked file")
    with archive.open("rb") as handle:
        raw = handle.read(ARCHIVE_SIZE + 1)
    if len(raw) != ARCHIVE_SIZE or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("ONNX Runtime archive size or SHA256 mismatch")
    # Parse only the immutable snapshot whose checksum was verified above.
    with zipfile.ZipFile(io.BytesIO(raw)) as tree:
        members, files, total = {}, set(), 0
        for item in tree.infolist():
            name = item.filename
            parts = name.rstrip("/").split("/")
            mode = item.external_attr >> 16
            if (len(members) >= 1000 or item.orig_filename != name or
                    not name or name.startswith("/") or
                    any(part in ("", ".", "..") for part in parts) or
                    any(char in name for char in ("\\", "\x00", ":")) or
                    name.rstrip("/") in members or item.flag_bits & 1 or
                    stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or
                    (stat.S_ISDIR(mode) and not item.is_dir()) or
                    (stat.S_ISREG(mode) and item.is_dir())):
                raise ValueError("unsafe, duplicate, or unsupported ZIP member")
            members[name.rstrip("/")] = item
            if not item.is_dir():
                files.add(name)
            total += item.file_size
            if item.file_size > 128 * 1024 * 1024 or total > 300 * 1024 * 1024:
                raise ValueError("ZIP extraction limit exceeded")
        for name in members:
            if any(str(parent) in files for parent in PurePosixPath(name).parents):
                raise ValueError("ZIP file/directory collision")
        if not set(MEMBERS).union({NUSPEC}).issubset(files):
            raise ValueError("required ONNX Runtime SDK member missing")
        metadata = ET.fromstring(tree.read(NUSPEC)).find("{*}metadata")
        identity = {"id": "Microsoft.ML.OnnxRuntime", "version": "1.17.3",
                    "authors": "Microsoft", "owners": "Microsoft"}
        if metadata is None or any(metadata.findtext("{*}" + key) != value
                                   for key, value in identity.items()):
            raise ValueError("ONNX Runtime package identity mismatch")
        selected = {destination: tree.read(source) for source, destination in MEMBERS.items()}
        if not re.search(rb"^#define ORT_API_VERSION 17\s*$",
                         selected["include/onnxruntime_c_api.h"], re.M):
            raise ValueError("ONNX Runtime API version mismatch")
    output.mkdir(mode=0o700)
    for name, content in selected.items():
        target = output / name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
    (output / "lib/libonnxruntime.so").symlink_to("libonnxruntime.so.1.17.3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
