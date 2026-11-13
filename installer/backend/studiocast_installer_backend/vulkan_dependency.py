"""Pinned user-local ncnn SDK planning, acquisition and atomic activation.

The lock, builder and patch are part of the authenticated Installer component.
No caller-selected source can supply dependency tooling or download metadata.
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from .contracts import (BackendError, atomic_write, digest, file_sha256, load_json,
                        roots, validate_mutation_path)
from .packaged import model_component_root, model_module, release_module

LOCK_PATH = 'installer/dependencies/ncnn-vulkan-v1.json'
MARKER = '.studiocast-sdk.json'


def load_dependency() -> tuple[dict[str, Any], Path, str]:
    root = model_component_root()
    path = root / LOCK_PATH
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 16384 or
            path.stat().st_nlink != 1 or any(parent.is_symlink() for parent in path.parents)):
        raise BackendError('vulkan_sdk.lock_invalid', 'Packaged Vulkan SDK lock is unavailable')
    lock = load_json(path)
    expected = {'schema_version', 'dependency', 'replay_abi', 'build_type', 'jobs',
                'build_reserve_bytes', 'builder', 'patch', 'sources'}
    if (set(lock) != expected or type(lock['schema_version']) is not int or lock['schema_version'] != 1 or
            lock['dependency'] != 'studiocast-ncnn-vulkan' or lock['replay_abi'] != 2 or
            lock['build_type'] != 'Release' or type(lock['jobs']) is not int or lock['jobs'] != 4 or
            type(lock['build_reserve_bytes']) is not int or lock['build_reserve_bytes'] != 2_000_000_000 or
            not isinstance(lock['sources'], list) or len(lock['sources']) != 2):
        raise BackendError('vulkan_sdk.lock_invalid', 'Unsupported Vulkan SDK lock')
    for key, relative in (('builder', 'scripts/build_ncnn_vulkan_dependency.py'),
                          ('patch', 'cmake/patches/ncnn-studiocast-replay.patch')):
        item = lock[key]
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256'} or item['path'] != relative or
                not re.fullmatch('[0-9a-f]{64}', str(item['sha256']))):
            raise BackendError('vulkan_sdk.lock_invalid', 'Invalid packaged SDK tooling binding')
        tool = root / relative
        if (tool.is_symlink() or not tool.is_file() or
                any(parent.is_symlink() for parent in tool.parents) or
                file_sha256(tool) != 'sha256:' + item['sha256']):
            raise BackendError('vulkan_sdk.tooling_changed', 'Packaged Vulkan SDK tooling failed integrity verification')
    identities = (('Tencent/ncnn', '24ea07e87f544a3fbc59980b57885a92c75f2d24',
                   '2bb389d15ef84498c9562cf5362938d0e25530aecefff70f4e3e04d97a637245', 15108296),
                  ('KhronosGroup/glslang', '5017667d5a5300dff6aab57db1931a495ff9a10c',
                   '1b5aac72a71c191527bb09f40ed6c9c71c8a4ed19e964063b6bf66f5586a5673', 4775739))
    for source, (project, revision, sha, size) in zip(lock['sources'], identities):
        expected_source = {'project': project, 'revision': revision, 'sha256': sha, 'size_bytes': size,
                           'url': f'https://codeload.github.com/{project}/tar.gz/{revision}',
                           'filename': f'{project.split("/")[-1]}-{revision}.tar.gz'}
        if source != expected_source:
            raise BackendError('vulkan_sdk.lock_invalid', 'SDK source identity is not pinned')
    return lock, root, file_sha256(path)


def dependency_inputs(offline: bool) -> dict[str, Any]:
    lock, _, lock_hash = load_dependency()
    fingerprint = digest(lock).split(':')[1]
    cache = roots()['cache'] / 'dependencies/ncnn-vulkan'
    destination = cache / fingerprint
    for path in (cache, destination):
        validate_mutation_path(path, [roots()['cache']])
        if path.is_symlink() or (path.exists() and
                (not path.is_dir() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022)):
            raise BackendError('vulkan_sdk.unsafe_path', 'SDK cache must be an owned directory')
    return {'lock_sha256': lock_hash, 'lock': copy.deepcopy(lock), 'fingerprint': fingerprint,
            'cache_root': str(cache), 'destination': str(destination),
            'prefix': str(destination / 'prefix'), 'offline': offline}


def dependency_downloads(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{'artifact_id': 'vulkan-sdk:' + source['project'], 'dependency': 'ncnn-vulkan',
             'filename': source['filename'], 'url': source['url'], 'size': source['size_bytes'],
             'size_status': 'known', 'sha256': 'sha256:' + source['sha256'], 'signature': None,
             'provenance': {'project': source['project'], 'revision': source['revision']},
             'license': {'id': 'BSD-3-Clause' if source['project'] == 'Tencent/ncnn' else 'LicenseRef-glslang',
                         'name': 'Full upstream license retained with SDK and runtime',
                         'url': f"https://github.com/{source['project']}/blob/{source['revision']}/LICENSE.txt"},
             'cache_path': str(Path(inputs['cache_root']) / 'archives/sha256' / source['sha256']),
             'destination': inputs['prefix'], 'required_for_core': True}
            for source in inputs['lock']['sources']]


def validate_inputs(inputs: dict[str, Any]) -> None:
    if type(inputs.get('offline')) is not bool or inputs != dependency_inputs(inputs['offline']):
        raise BackendError('vulkan_sdk.plan_changed', 'Reviewed SDK bindings differ from the packaged dependency')


def validate_plan(plan: dict[str, Any]) -> None:
    operations = plan.get('operations', [])
    sdk = [op for op in operations if op.get('kind') == 'vulkan_sdk_provision']
    configure = [op for op in operations if op.get('kind') == 'build_configure']
    enabled = plan.get('desired_state', {}).get('features', {}).get('open_vulkan') is True
    if len(configure) > 1 or len(sdk) != (1 if enabled and configure else 0):
        raise BackendError('vulkan_sdk.plan_changed', 'Vulkan configuration requires exactly one reviewed SDK operation')
    rows = [row for row in plan.get('downloads', []) if row.get('dependency') == 'ncnn-vulkan' or
            str(row.get('artifact_id', '')).startswith('vulkan-sdk:')]
    if not sdk:
        if rows:
            raise BackendError('vulkan_sdk.plan_changed', 'SDK downloads have no reviewed operation')
        return
    op = sdk[0]
    validate_inputs(op['inputs'])
    destination = Path(op['inputs']['destination'])
    if destination.exists():
        verify_prefix(destination, op['inputs'])
    if (op.get('id') != 'vulkan.sdk.provision' or op.get('privilege') != 'user' or
            op.get('failure_policy') != 'abort_rollback' or
            rows != dependency_downloads(op['inputs']) or
            operations.index(op) >= operations.index(configure[0])):
        raise BackendError('vulkan_sdk.plan_changed', 'Reviewed SDK operation or downloads changed')
    if (plan.get('disk', {}).get('base_required_bytes', -1) < op['inputs']['lock']['build_reserve_bytes'] or
            configure[0]['inputs'].get('ncnn_prefix') != op['inputs']['prefix']):
        raise BackendError('vulkan_sdk.plan_changed', 'Configure does not bind the reviewed SDK prefix')


def prefix_inventory(prefix: Path) -> dict[str, Any]:
    if prefix.is_symlink() or not prefix.is_dir():
        raise BackendError('vulkan_sdk.prefix_invalid', 'SDK prefix is not a regular directory')
    inventory = {}
    total = 0
    for path in sorted(prefix.rglob('*')):
        mode = path.lstat()
        if (path.is_symlink() or mode.st_uid != os.getuid() or
                not (stat.S_ISDIR(mode.st_mode) or stat.S_ISREG(mode.st_mode))):
            raise BackendError('vulkan_sdk.prefix_invalid', 'SDK prefix contains an unsafe entry')
        if path.is_dir():
            continue
        total += mode.st_size
        if len(inventory) >= 10000 or total > 1_000_000_000 or mode.st_nlink != 1:
            raise BackendError('vulkan_sdk.prefix_invalid', 'SDK prefix exceeds its integrity bounds')
        inventory[path.relative_to(prefix).as_posix()] = {'sha256': file_sha256(path), 'size': mode.st_size}
    required = ('lib/cmake/ncnn/ncnnConfig.cmake', 'lib/libncnn.a',
                'share/licenses/studiocast-ncnn/ncnn-LICENSE.txt',
                'share/licenses/studiocast-ncnn/glslang-LICENSE.txt',
                'share/licenses/studiocast-ncnn/dependency.json')
    if any(name not in inventory or inventory[name]['size'] == 0 for name in required):
        raise BackendError('vulkan_sdk.prefix_invalid', 'SDK prefix is incomplete')
    return inventory


def expected_receipt(inputs: dict[str, Any]) -> dict[str, Any]:
    lock = inputs['lock']
    return {'upstream_revision': lock['sources'][0]['revision'],
            'glslang_revision': lock['sources'][1]['revision'], 'replay_abi': lock['replay_abi'],
            'patch_sha256': lock['patch']['sha256']}


def verify_prefix(destination: Path, inputs: dict[str, Any]) -> None:
    validate_mutation_path(destination / MARKER, [roots()['cache']])
    marker = destination / MARKER
    if (marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 4_000_000 or
            marker.stat().st_nlink != 1 or marker.stat().st_uid != os.getuid()):
        raise BackendError('vulkan_sdk.prefix_invalid', 'SDK ownership receipt is missing or unsafe')
    receipt = load_json(marker)
    prefix = destination / 'prefix'
    if (set(receipt) != {'schema_version', 'fingerprint', 'lock_sha256', 'files'} or
            receipt['schema_version'] != 1 or receipt['fingerprint'] != inputs['fingerprint'] or
            receipt['lock_sha256'] != inputs['lock_sha256'] or receipt['files'] != prefix_inventory(prefix) or
            load_json(prefix / 'share/licenses/studiocast-ncnn/dependency.json') != expected_receipt(inputs)):
        raise BackendError('vulkan_sdk.prefix_invalid', 'SDK receipt or installed bytes failed verification')


def verify_dlib_configuration(build: Path) -> None:
    cache = build / 'CMakeCache.txt'
    validate_mutation_path(cache, [roots()['cache']])
    if cache.is_symlink() or not cache.is_file() or cache.stat().st_size > 4_000_000:
        raise BackendError('vulkan_sdk.dlib_configuration_missing', 'Configure did not produce a valid dependency receipt')
    values = {}
    for line in cache.read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.startswith(('//', '#')):
            key, value = line.split('=', 1)
            key = key.split(':', 1)[0]
            if key in {'STUDIOCAST_ENABLE_DLIB', 'STUDIOCAST_DLIB_AVAILABLE'}:
                if key in values:
                    raise BackendError('vulkan_sdk.dlib_configuration_missing', 'Configure returned duplicate dlib receipts')
                values[key] = value.upper()
    if any(values.get(key) not in {'TRUE', 'ON', '1'}
           for key in ('STUDIOCAST_ENABLE_DLIB', 'STUDIOCAST_DLIB_AVAILABLE')):
        raise BackendError('vulkan_sdk.dlib_unavailable',
                           'Vulkan Eye Contact requires usable dlib. Install libdlib-dev, then retry with a fresh build cache.')


def provision(inputs: dict[str, Any], run_process: Any, dry_run: bool = False,
              transport: Any = None) -> None:
    validate_inputs(inputs)
    _, root, _ = load_dependency()
    destination = Path(inputs['destination'])
    if destination.exists():
        verify_prefix(destination, inputs)
        return
    if dry_run:
        return
    cache = Path(inputs['cache_root'])
    validate_mutation_path(cache, [roots()['cache']])
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    module = model_module()
    rc = release_module()
    # Reuse the existing resumable/hash-verified artifact cache. Its transport
    # adapter uses the release HTTPS redirect/range checks and enforces size.
    class SourceTransport:
        def __init__(self, source: dict[str, Any]):
            self.source = source
        def fetch(self, relative: str, partial: Path, offset: int) -> int:
            if partial.exists() and (partial.is_symlink() or partial.stat().st_nlink != 1):
                raise module.TransactionError('unsafe_cache', 'SDK partial is unsafe')
            if offset > self.source['size_bytes']:
                partial.unlink()
                offset = 0
            try:
                (transport or rc.UrlTransport(maximum_bytes=self.source["size_bytes"])).append(self.source['url'], partial, offset)
            except rc.ReleaseChannelError as exc:
                raise module.TransactionError(exc.code, str(exc)) from exc
            if partial.stat().st_size > self.source['size_bytes']:
                partial.unlink()
                raise module.TransactionError('checksum_mismatch', 'SDK download exceeds pinned size')
            return partial.stat().st_size - offset
    archives = []
    try:
        for source in inputs['lock']['sources']:
            archive_cache = cache / 'archives'
            validate_mutation_path(archive_cache / 'sha256' / source['sha256'], [roots()['cache']])
            for name in (source['sha256'], source['sha256'] + '.part'):
                existing = archive_cache / 'sha256' / name
                if existing.exists() and (existing.is_symlink() or existing.stat().st_nlink != 1):
                    raise BackendError('vulkan_sdk.unsafe_path', 'SDK archive cache is unsafe')
            provider = module.ArtifactProvider(archive_cache, SourceTransport(source),
                                              offline=inputs['offline'], retries=2)
            artifact = {'sha256': source['sha256'], 'size_bytes': source['size_bytes'],
                        'source_paths': [source['filename']]}
            archives.append(provider.acquire(artifact, None, force=False).path)
    except module.TransactionError as exc:
        raise BackendError('vulkan_sdk.' + exc.reason_code, str(exc)) from exc
    temporary = Path(tempfile.mkdtemp(prefix='.build-', dir=cache))
    output = temporary / 'output'
    try:
        run_process([sys.executable, '-I', str(root / inputs['lock']['builder']['path']),
                     '--ncnn-archive', str(archives[0]), '--glslang-archive', str(archives[1]),
                     '--output', str(output), '--jobs', str(inputs['lock']['jobs'])])
        prefix = output / 'prefix'
        inventory = prefix_inventory(prefix)
        if load_json(prefix / 'share/licenses/studiocast-ncnn/dependency.json') != expected_receipt(inputs):
            raise BackendError('vulkan_sdk.prefix_invalid', 'Built SDK dependency receipt differs from the reviewed lock')
        atomic_write(output / MARKER, {'schema_version': 1, 'fingerprint': inputs['fingerprint'],
                                     'lock_sha256': inputs['lock_sha256'], 'files': inventory})
        verify_prefix(output, inputs)
        # Keep immutable installed products; build/source scratch is not reused.
        shutil.rmtree(output / 'source')
        shutil.rmtree(output / 'build')
        if destination.exists() or destination.is_symlink():
            raise BackendError('vulkan_sdk.activation_conflict', 'SDK destination appeared during the build')
        output.chmod(0o700)
        os.rename(output, destination)
    finally:
        shutil.rmtree(temporary)
