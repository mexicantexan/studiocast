#!/usr/bin/env python3
"""Hermetic SDK acquisition/build/plan boundary regressions; never builds on host."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'installer/backend'))
sys.path.insert(0, str(Path(__file__).parent))
from studiocast_installer_backend import vulkan_dependency as sdk
from studiocast_installer_backend import executor, plan, contracts
from installer_backend_core_tests import base_facts
from studiocast_installer_backend.recommendation import recommend
spec = importlib.util.spec_from_file_location('ncnn_builder', ROOT / 'scripts/build_ncnn_vulkan_dependency.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='studiocast-sdk-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / 'home with spaces;$(literal)'
        self.home.mkdir()
        self.env = mock.patch.dict(os.environ, {'HOME': str(self.home),
            'XDG_CACHE_HOME': str(self.home / '.cache'), 'XDG_DATA_HOME': str(self.home / '.data'),
            'XDG_STATE_HOME': str(self.home / '.state'), 'XDG_CONFIG_HOME': str(self.home / '.config')})
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_plan(self, enabled=True, **overrides):
        facts = base_facts()
        facts["host_fingerprint"] = contracts.digest(facts)
        args = argparse.Namespace(workflow='install', route='advanced', open_vulkan=enabled,
                                  models=False, v4l2=False, service=False, **overrides)
        rec = recommend(facts, 'install', None, route='advanced')
        return plan.create_plan(args, facts, rec)

    def populate_prefix(self, output, inputs):
        prefix = output / 'prefix'
        for name in ('lib/cmake/ncnn/ncnnConfig.cmake', 'lib/libncnn.a',
                     'share/licenses/studiocast-ncnn/ncnn-LICENSE.txt',
                     'share/licenses/studiocast-ncnn/glslang-LICENSE.txt'):
            path = prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture ' + name)
        receipt = prefix / 'share/licenses/studiocast-ncnn/dependency.json'
        receipt.write_text(json.dumps(sdk.expected_receipt(inputs)))
        (output / 'source').mkdir()
        (output / 'build').mkdir()

    def fake_builder(self, inputs, calls, fail=False):
        def run(argv):
            calls.append(argv)
            if fail:
                raise contracts.BackendError('fixture.failed', 'fixture builder failure')
            self.populate_prefix(Path(argv[argv.index('--output') + 1]), inputs)
        return run

    def test_plan_on_off_and_exact_download_disk_binding(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                reviewed = self.make_plan(enabled)
                sdk.validate_plan(reviewed)
                configure = next(op for op in reviewed['operations'] if op['kind'] == 'build_configure')
                executor.validate_operation_contract(reviewed, configure)
                self.assertEqual(configure['inputs']['cmake']['STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING'], enabled)
                rows = [row for row in reviewed['downloads'] if row.get('dependency')]
                self.assertEqual(len(rows), 2 if enabled else 0)
                self.assertEqual(sum(row['size'] for row in rows), 19884035 if enabled else 0)
                self.assertEqual(reviewed['disk']['base_required_bytes'], 2000000003 if enabled else 3)
                self.assertFalse(any('pack_id' in row for row in rows))
                self.assertEqual(configure['inputs']['ncnn_prefix'] is not None, enabled)
                executor.validate_model_plan_contract(reviewed)

    def test_tampered_plan_prefix_lock_flag_and_downloads_rejected(self):
        reviewed = self.make_plan()
        mutations = ('prefix', 'lock', 'download', 'remove_sdk', 'flag', 'extra_flag')
        for mutation in mutations:
            candidate = copy.deepcopy(reviewed)
            op = next(op for op in candidate['operations'] if op['kind'] == 'vulkan_sdk_provision')
            configure = next(op for op in candidate['operations'] if op['kind'] == 'build_configure')
            if mutation == 'prefix': configure['inputs']['ncnn_prefix'] = '/tmp/attacker'
            if mutation == 'lock': op['inputs']['lock']['sources'][0]['url'] = 'https://attacker.invalid/sdk'
            if mutation == 'download': candidate['downloads'][-1]['size'] += 1
            if mutation == 'remove_sdk': candidate['operations'].remove(op)
            if mutation == 'flag': configure['inputs']['cmake']['STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING'] = False
            if mutation == 'extra_flag': configure['inputs']['cmake']['CMAKE_TOOLCHAIN_FILE'] = True
            with self.subTest(mutation=mutation), self.assertRaises(contracts.BackendError):
                for operation in candidate['operations']:
                    executor.validate_operation_contract(candidate, operation)
                sdk.validate_plan(candidate)

    def test_dry_run_has_no_download_build_or_mutation(self):
        inputs = sdk.dependency_inputs(True)
        calls = []
        sdk.provision(inputs, calls.append, dry_run=True)
        self.assertEqual(calls, [])
        self.assertFalse(Path(inputs['cache_root']).exists())

    def test_packaged_tooling_tamper_is_rejected(self):
        packaged = self.root / 'component'
        for relative in (sdk.LOCK_PATH, 'scripts/build_ncnn_vulkan_dependency.py',
                         'cmake/patches/ncnn-studiocast-replay.patch'):
            target = packaged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        with mock.patch.object(sdk, 'model_component_root', return_value=packaged):
            sdk.load_dependency()
            for relative in ('scripts/build_ncnn_vulkan_dependency.py', 'cmake/patches/ncnn-studiocast-replay.patch'):
                target = packaged / relative
                old = target.read_bytes()
                target.write_bytes(old + b'\n# tamper\n')
                with self.assertRaisesRegex(contracts.BackendError, 'integrity'):
                    sdk.load_dependency()
                target.write_bytes(old)
            lock = packaged / sdk.LOCK_PATH
            data = json.loads(lock.read_text())
            data['sources'][0]['size_bytes'] += 1
            lock.write_text(json.dumps(data))
            with self.assertRaises(contracts.BackendError):
                sdk.load_dependency()

    def test_atomic_activation_offline_reuse_and_prefix_mutation(self):
        inputs = sdk.dependency_inputs(True)
        module = sdk.model_module()
        paths = []
        for i in range(2):
            path = self.root / f'archive{i}'
            path.write_bytes(b'fixture')
            paths.append(path)
        # Inject a provider with tiny acquired paths; the real helper separately
        # enforces source archive bytes before any extraction or tool execution.
        with mock.patch.object(sdk, 'model_module', return_value=module), mock.patch.object(
                module.ArtifactProvider, 'acquire', side_effect=[mock.Mock(path=p) for p in paths]):
            calls = []
            sdk.provision(inputs, self.fake_builder(inputs, calls))
        destination = Path(inputs['destination'])
        self.assertTrue(destination.is_dir())
        self.assertEqual(len(calls), 1)
        self.assertIn('--ncnn-archive', calls[0])
        self.assertEqual(calls[0][0:2], [sys.executable, '-I'])
        sdk.verify_prefix(destination, inputs)
        sdk.provision(inputs, lambda argv: self.fail('cache reuse ran a build'))
        (destination / 'prefix/lib/libncnn.a').write_text('tampered')
        with self.assertRaises(contracts.BackendError):
            sdk.provision(inputs, lambda argv: self.fail('tampered prefix ran a build'))

    def test_build_failure_has_no_activation(self):
        inputs = sdk.dependency_inputs(True)
        module = sdk.model_module()
        archive = self.root / 'fake'
        archive.write_bytes(b'fixture')
        with mock.patch.object(sdk, 'model_module', return_value=module), mock.patch.object(
                module.ArtifactProvider, 'acquire', return_value=mock.Mock(path=archive)):
            with self.assertRaises(contracts.BackendError):
                sdk.provision(inputs, self.fake_builder(inputs, [], fail=True))
        self.assertFalse(Path(inputs['destination']).exists())
        self.assertEqual(list(Path(inputs['cache_root']).glob('.build-*')), [])

    def test_cache_symlink_and_hardlink_rejected_before_transport(self):
        inputs = sdk.dependency_inputs(True)
        cache = Path(inputs['cache_root']) / 'archives/sha256'
        cache.mkdir(parents=True)
        source = self.root / 'existing'
        source.write_bytes(b'fixture')
        target = cache / inputs['lock']['sources'][0]['sha256']
        for kind in ('symlink', 'hardlink'):
            if kind == 'symlink': target.symlink_to(source)
            else: os.link(source, target)
            with self.assertRaises(contracts.BackendError):
                sdk.provision(inputs, lambda argv: self.fail('unsafe cache ran a build'))
            self.assertEqual(source.read_bytes(), b'fixture')
            target.unlink()

    def test_cache_parent_symlink_and_writable_sdk_root_fail_closed(self):
        cache = contracts.roots()['cache']
        cache.mkdir(parents=True)
        outside = self.root / 'outside'
        outside.mkdir()
        (cache / 'dependencies').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(contracts.BackendError):
            sdk.dependency_inputs(False)
        self.assertEqual(list(outside.iterdir()), [])
        (cache / 'dependencies').unlink()
        sdk_cache = cache / 'dependencies/ncnn-vulkan'
        sdk_cache.mkdir(parents=True)
        sdk_cache.chmod(0o777)
        with self.assertRaises(contracts.BackendError):
            sdk.dependency_inputs(False)

    def test_archive_hash_fails_before_output_or_process(self):
        archive = self.root / 'bad.tgz'
        archive.write_bytes(b'bad')
        output = self.root / 'out'
        proc = __import__('subprocess').run([sys.executable, str(ROOT / 'scripts/build_ncnn_vulkan_dependency.py'),
            '--ncnn-archive', str(archive), '--glslang-archive', str(archive), '--output', str(output)],
            text=True, capture_output=True, env=os.environ.copy())
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('checksum or size mismatch', proc.stderr)
        self.assertFalse(output.exists())

    def test_archive_rejects_traversal_links_duplicate_special_and_limits(self):
        for bad in ('../escape', '/abs', 'root/../escape', 'root/link', 'hardlink', 'duplicate', 'special', 'collision'):
            with self.subTest(bad=bad):
                raw = io.BytesIO()
                with tarfile.open(fileobj=raw, mode='w:gz') as archive:
                    info = tarfile.TarInfo('root/file')
                    info.size = 1
                    archive.addfile(info, io.BytesIO(b'x'))
                    info = tarfile.TarInfo('root/file' if bad == 'duplicate' else bad)
                    if bad == 'root/link': info.type, info.linkname = tarfile.SYMTYPE, '/escape'
                    elif bad == 'hardlink': info.name, info.type, info.linkname = 'root/hard', tarfile.LNKTYPE, 'root/file'
                    elif bad == 'special': info.name, info.type = 'root/device', tarfile.CHRTYPE
                    elif bad == 'collision': info.name = 'root/file/child'
                    archive.addfile(info)
                path = self.root / 'test.tgz'
                data = raw.getvalue()
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    builder.verified_archive(path, hashlib.sha256(data).hexdigest(), len(data), 'root')

    def test_safe_archive_extracts_only_verified_single_top(self):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w:gz') as archive:
            item = tarfile.TarInfo('root/folder/file')
            item.size = 4
            archive.addfile(item, io.BytesIO(b'data'))
        data = raw.getvalue()
        path = self.root / 'safe.tgz'
        path.write_bytes(data)
        tree, members = builder.verified_archive(path, hashlib.sha256(data).hexdigest(), len(data), 'root')
        builder.extract_verified(tree, members, self.root / 'extracted')
        self.assertEqual((self.root / 'extracted/folder/file').read_bytes(), b'data')

    def test_modify_update_preserve_prior_vulkan_feature(self):
        for workflow in ('modify', 'update'):
            facts = base_facts()
            facts['host_fingerprint'] = contracts.digest(facts)
            prior = {'features': {'open_cuda': False, 'open_audio': False, 'open_vulkan': True},
                     'build_type': 'Release'}
            rec = recommend(facts, workflow, prior, route='custom')
            args = argparse.Namespace(workflow=workflow, models=False, v4l2=False, service=False)
            reviewed = plan.create_plan(args, facts, rec)
            self.assertTrue(reviewed['desired_state']['features']['open_vulkan'])
            sdk.validate_plan(reviewed)

    def test_missing_or_false_dlib_configure_receipt_aborts(self):
        build = contracts.roots()['cache'] / 'build'
        build.mkdir(parents=True)
        for content in ('', 'STUDIOCAST_ENABLE_DLIB:BOOL=OFF\nSTUDIOCAST_DLIB_AVAILABLE:INTERNAL=TRUE',
                        'STUDIOCAST_ENABLE_DLIB:BOOL=ON\nSTUDIOCAST_DLIB_AVAILABLE:INTERNAL=FALSE'):
            (build / 'CMakeCache.txt').write_text(content)
            with self.assertRaises(contracts.BackendError):
                sdk.verify_dlib_configuration(build)
        (build / 'CMakeCache.txt').write_text('STUDIOCAST_ENABLE_DLIB:BOOL=ON\nSTUDIOCAST_DLIB_AVAILABLE:INTERNAL=TRUE')
        sdk.verify_dlib_configuration(build)

    def test_missing_sdk_prerequisites_block_or_use_existing_helper_allowlist(self):
        for managed in (False, True):
            facts = base_facts()
            facts['host_fingerprint'] = contracts.digest(facts)
            facts['toolchain']['missing_vulkan_sdk_dependencies'] = ['git', 'dlib']
            rec = recommend(facts, 'install', None, route='advanced')
            args = argparse.Namespace(workflow='install', open_vulkan=True, models=False,
                                      v4l2=False, service=False, deps=managed)
            reviewed = plan.create_plan(args, facts, rec)
            blockers = [row['code'] for row in reviewed['blockers']]
            self.assertEqual('vulkan_sdk.build_dependencies_missing' in blockers, not managed)
            package_ops = [op for op in reviewed['operations'] if op['id'] == 'dependencies.packages.ensure']
            if managed:
                self.assertEqual(package_ops[0]['inputs']['packages'], ['git', 'libdlib-dev'])
            else:
                self.assertEqual(package_ops, [])

    def test_configure_uses_literal_bound_prefix_argument(self):
        reviewed = self.make_plan()
        configure = next(op for op in reviewed['operations'] if op['kind'] == 'build_configure')
        engine = executor.Executor(reviewed, dry_run=True)
        calls = []
        engine.run_process = calls.append
        engine.execute_one(configure)
        self.assertEqual(len(calls), 1)
        self.assertIn('-DSTUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=ON', calls[0])
        self.assertIn('-Dncnn_DIR=' + configure['inputs']['ncnn_prefix'] + '/lib/cmake/ncnn', calls[0])
        self.assertIn(';$(literal)', calls[0][-1])
        self.assertFalse((self.home / '.cache').exists())

    def test_packaged_installer_layout_loads_exact_sdk_surface(self):
        component = self.root / 'staged component'
        installer = component / 'installer'
        shutil.copytree(ROOT / 'installer/backend/studiocast_installer_backend',
                        installer / 'studiocast_installer_backend',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for relative in (sdk.LOCK_PATH, 'scripts/build_ncnn_vulkan_dependency.py',
                         'cmake/patches/ncnn-studiocast-replay.patch',
                         'installer/models/curated-model-catalog-v1.json'):
            target = component / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        result = __import__('subprocess').run([sys.executable, '-I', '-c',
            'import sys;sys.path.insert(0,sys.argv[1]);'
            'from studiocast_installer_backend.vulkan_dependency import load_dependency;'
            'print(load_dependency()[1])', str(installer)],
            text=True, capture_output=True, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(component))

    def test_real_artifact_provider_offline_and_source_failure_integrity(self):
        module = sdk.model_module()
        payload = b'verified tiny source archive fixture'
        sha = hashlib.sha256(payload).hexdigest()
        artifact = {'sha256': sha, 'size_bytes': len(payload), 'source_paths': ['source.tgz']}
        provider = module.ArtifactProvider(self.root / 'provider-cache', None, offline=True, retries=0)
        with self.assertRaises(module.TransactionError):
            provider.acquire(artifact, None, force=False)
        cached = provider.artifact_cache / sha
        cached.write_bytes(payload)
        self.assertEqual(provider.acquire(artifact, None, force=False).path, cached)
        class FailingTransport:
            def fetch(self, relative, partial, offset):
                partial.write_bytes(b'corrupt')
                raise module.TransactionError('source_unavailable', 'fixture source failure')
        provider = module.ArtifactProvider(self.root / 'provider-cache', FailingTransport(), offline=False, retries=0)
        with self.assertRaises(module.TransactionError):
            provider.acquire(artifact, None, force=True)
        self.assertEqual(cached.read_bytes(), payload)
        self.assertEqual(provider.acquire(artifact, None, force=False).path, cached)

    def test_bounded_https_transport_rejects_before_partial_exceeds_pin(self):
        rc = sdk.release_module()
        class Response(io.BytesIO):
            status = 200
            def geturl(self): return 'https://example.invalid/archive'
        partial = self.root / 'download.part'
        response = Response(b'123456789')
        transport = rc.UrlTransport(opener=lambda *a, **kw: response, maximum_bytes=4)
        with self.assertRaises(rc.ReleaseChannelError):
            transport.append('https://example.invalid/archive', partial, 0)
        self.assertLessEqual(partial.stat().st_size, 4)
        partial.write_bytes(b'12345')
        response = Response(b'6789')
        response.status, response.headers = 206, {'Content-Range': 'bytes 5-8/9'}
        with self.assertRaises(rc.ReleaseChannelError):
            rc.UrlTransport(opener=lambda *a, **kw: response, maximum_bytes=4).append(
                'https://example.invalid/archive', partial, 5)
        self.assertEqual(partial.read_bytes(), b'12345')


if __name__ == '__main__':
    unittest.main()
