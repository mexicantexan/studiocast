#!/usr/bin/env python3
"""Build the reviewed ncnn replay dependency into an explicit isolated prefix.

No network access or package installation. Supply an exact upstream checkout
with its pinned glslang submodule already available. All copied sources, build
products and prefix files remain below --output (which must not exist).
"""
import argparse
import hashlib
import json
import pathlib
import io
import tarfile
import subprocess

REVISION = '24ea07e87f544a3fbc59980b57885a92c75f2d24'
GLSLANG_REVISION = '5017667d5a5300dff6aab57db1931a495ff9a10c'

def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or args.jobs < 1:
        parser.error('--output must not exist and --jobs must be positive')
    for path, revision in ((source, REVISION), (source/'glslang', GLSLANG_REVISION)):
        if run('git', '-C', str(path), 'rev-parse', 'HEAD', capture_output=True).stdout.strip() != revision:
            parser.error(f'{path} must be pinned to {revision}')
    patch = pathlib.Path(__file__).resolve().parents[1]/'cmake/patches/ncnn-studiocast-replay.patch'
    output.mkdir(parents=True)
    destination = output/'source'
    destination.mkdir()
    for repository, revision, target in ((source, REVISION, destination),
            (source/'glslang', GLSLANG_REVISION, destination/'glslang')):
        target.mkdir(exist_ok=True)
        archive = subprocess.run(['git', '-C', str(repository), 'archive', revision],
                                 check=True, stdout=subprocess.PIPE).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
            tree.extractall(target, filter='data')
    run('git', 'apply', '--unidiff-zero', '--check', str(patch), cwd=destination)
    run('git', 'apply', '--unidiff-zero', str(patch), cwd=destination)
    receipt = {'upstream_revision': REVISION, 'glslang_revision': GLSLANG_REVISION,
               'replay_abi': 2, 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest()}
    (output/'dependency.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if args.prepare_only:
        return
    run('cmake', '-S', str(destination), '-B', str(output/'build'), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DNCNN_VULKAN=ON', '-DNCNN_SIMPLEVK=ON',
        '-DNCNN_SHARED_LIB=OFF', '-DNCNN_BUILD_TOOLS=OFF', '-DNCNN_BUILD_EXAMPLES=OFF',
        '-DNCNN_BUILD_BENCHMARK=OFF', '-DNCNN_BUILD_TESTS=OFF', '-DNCNN_COVERAGE=OFF',
        f'-DCMAKE_INSTALL_PREFIX={output / "prefix"}')
    run('cmake', '--build', str(output/'build'), '-j', str(args.jobs))
    run('cmake', '--install', str(output/'build'))
    license_root = output/'prefix/share/licenses/studiocast-ncnn'
    license_root.mkdir(parents=True)
    import shutil
    shutil.copyfile(destination/'LICENSE.txt', license_root/'ncnn-LICENSE.txt')
    shutil.copyfile(destination/'glslang/LICENSE.txt', license_root/'glslang-LICENSE.txt')
    shutil.copyfile(output/'dependency.json', license_root/'dependency.json')
    print(output/'prefix/lib/cmake/ncnn')

if __name__ == '__main__':
    main()
