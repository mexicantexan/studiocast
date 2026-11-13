#!/usr/bin/env python3
"""Build the reviewed ncnn replay dependency in a new isolated output.

Accept an exact local Git checkout or the two pinned source archives. No network
or system installation is performed. All products stay beneath --output.
"""
import argparse
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import tarfile

REVISION = '24ea07e87f544a3fbc59980b57885a92c75f2d24'
GLSLANG_REVISION = '5017667d5a5300dff6aab57db1931a495ff9a10c'
ARCHIVES = (
    ('ncnn', REVISION, '2bb389d15ef84498c9562cf5362938d0e25530aecefff70f4e3e04d97a637245', 15108296),
    ('glslang', GLSLANG_REVISION, '1b5aac72a71c191527bb09f40ed6c9c71c8a4ed19e964063b6bf66f5586a5673', 4775739),
)


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def verified_archive(path, expected_hash, expected_size, top):
    """Snapshot verified compressed bytes and validate every member before writes."""
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('source archive must be a regular, non-linked file')
    with path.open('rb') as handle:
        raw = handle.read(expected_size + 1)
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('source archive checksum or size mismatch')
    tree = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
    seen, members, total = set(), [], 0
    for member in tree:
        if len(members) >= 10000:
            raise ValueError('source archive has too many entries')
        parts = pathlib.PurePosixPath(member.name).parts
        if (not parts or parts[0] != top or member.name.startswith('/') or
                any(part in ('.', '..') for part in member.name.rstrip('/').split('/')) or
                '\\' in member.name or '\x00' in member.name or
                not (member.isdir() or member.isfile()) or member.name.rstrip('/') in seen):
            raise ValueError('unsafe or duplicate source archive member')
        seen.add(member.name.rstrip('/'))
        total += member.size
        if member.size < 0 or member.size > 128 * 1024 * 1024 or total > 160 * 1024 * 1024:
            raise ValueError('source archive extraction limit exceeded')
        members.append(member)
    # A regular file cannot be an ancestor of another entry.
    files = {m.name.rstrip('/') for m in members if m.isfile()}
    for member in members:
        if any(str(parent) in files for parent in pathlib.PurePosixPath(member.name).parents):
            raise ValueError('source archive file/directory collision')
    return tree, members


def extract_verified(tree, members, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for member in members:
        relative = pathlib.PurePosixPath(member.name).parts[1:]
        if not relative:
            if not member.isdir():
                raise ValueError('source archive top must be a directory')
            continue
        target = destination.joinpath(*relative)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tree.extractfile(member) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    tree.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=pathlib.Path)
    parser.add_argument('--ncnn-archive', type=pathlib.Path)
    parser.add_argument('--glslang-archive', type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or '..' in output.parts or not 1 <= args.jobs <= 64:
        parser.error('--output must be new, and --jobs must be between 1 and 64')
    if any(parent.is_symlink() for parent in output.parents):
        parser.error('--output may not have symlink parents')
    archive_mode = args.ncnn_archive is not None and args.glslang_archive is not None
    if bool(args.source) == archive_mode or (bool(args.ncnn_archive) != bool(args.glslang_archive)):
        parser.error('supply --source or both pinned archives')
    verified = []
    if archive_mode:
        for path, (name, revision, sha, size) in zip((args.ncnn_archive, args.glslang_archive), ARCHIVES):
            verified.append(verified_archive(path, sha, size, f'{name}-{revision}'))
    else:
        source = args.source.resolve()
        for path, revision in ((source, REVISION), (source/'glslang', GLSLANG_REVISION)):
            if run('git', '-C', str(path), 'rev-parse', 'HEAD', capture_output=True).stdout.strip() != revision:
                parser.error(f'{path} must be pinned to {revision}')
    patch = pathlib.Path(__file__).resolve().parents[1]/'cmake/patches/ncnn-studiocast-replay.patch'
    output.mkdir(parents=True)
    destination = output/'source'
    destination.mkdir()
    if archive_mode:
        for (tree, members), target in zip(verified, (destination, destination/'glslang')):
            extract_verified(tree, members, target)
    else:
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
    shutil.copyfile(destination/'LICENSE.txt', license_root/'ncnn-LICENSE.txt')
    shutil.copyfile(destination/'glslang/LICENSE.txt', license_root/'glslang-LICENSE.txt')
    shutil.copyfile(output/'dependency.json', license_root/'dependency.json')
    print(output/'prefix/lib/cmake/ncnn')


if __name__ == '__main__':
    main()
