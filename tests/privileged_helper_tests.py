#!/usr/bin/python3
"""Hermetic tests for the separately packaged StudioCast system helper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = REPO_ROOT / "installer/privileged/studiocast_system_helper.py"
CLIENT_SOURCE = REPO_ROOT / "installer/privileged/client_contract.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper_mod = load_module("studiocast_system_helper", HELPER_SOURCE)
client_mod = load_module("studiocast_helper_client_contract", CLIENT_SOURCE)


class FakeExecutor:
    is_test_adapter = True

    def __init__(self, results=None):
        self.commands = []
        self.results = list(results or [])

    def run(self, argv, timeout_seconds=helper_mod.COMMAND_TIMEOUT_SECONDS):
        self.commands.append((list(argv), timeout_seconds))
        if self.results:
            return self.results.pop(0)
        return helper_mod.CommandResult(0)


class FakeSystem:
    is_test_adapter = True

    def __init__(self, *, base=("ubuntu", "24.04"), kernel="6.8.0-57-generic", conflict=False, busy=False):
        self.base = base
        self.kernel = kernel
        self.conflict = conflict
        self.busy = busy
        self.conflict_calls = []

    def base_os_release(self):
        return self.base

    def kernel_release(self):
        return self.kernel

    def device_conflicts(self, number, allow_loopback):
        self.conflict_calls.append((number, allow_loopback))
        return self.conflict

    def loopback_busy(self):
        return self.busy


def valid_request(operations=None):
    return {
        "schema_version": 1,
        "request_id": str(uuid.uuid4()),
        "transaction_id": str(uuid.uuid4()),
        "plan_digest": "sha256:" + "a" * 64,
        "policy_version": 1,
        "preconditions": {
            "base_os": "ubuntu",
            "base_release": "24.04",
            "kernel_release": "6.8.0-57-generic",
        },
        "operations": operations
        or [{"id": "packages", "type": "packages.ensure.v1", "packages": ["cmake"]}],
    }


def encoded(request):
    return json.dumps(request, separators=(",", ":")).encode()


class HelperHarness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="studiocast-helper-test-")
        self.root = Path(self.temp.name)
        (self.root / "modules-load.d").mkdir()
        (self.root / "modprobe.d").mkdir()
        self.executor = FakeExecutor()
        self.system = FakeSystem()
        self.store = helper_mod.PersistenceStore(
            self.root,
            test_mode=True,
            required_uid=os.getuid(),
            required_gid=os.getgid(),
        )
        self.helper = helper_mod.Helper(
            self.executor, self.system, self.store, test_mode=True
        )

    def tearDown(self):
        self.temp.cleanup()

    def handle(self, request):
        return self.helper.handle(encoded(request))


class ParserValidationTests(HelperHarness):
    def test_exact_allowlisted_packages_use_fixed_arrays(self):
        request = valid_request(
            [{
                "id": "packages",
                "type": "packages.ensure.v1",
                "packages": ["cmake", "v4l-utils", "linux-headers-6.8.0-57-generic"],
            }]
        )
        result, code = self.handle(request)
        self.assertEqual((code, result["state"]), (0, "succeeded"))
        self.assertEqual(
            [command for command, _ in self.executor.commands],
            [
                ["/usr/bin/apt-get", "update"],
                [
                    "/usr/bin/apt-get",
                    "install",
                    "-y",
                    "--",
                    "cmake",
                    "v4l-utils",
                    "linux-headers-6.8.0-57-generic",
                ],
            ],
        )
        self.assertTrue(all(isinstance(command, list) for command, _ in self.executor.commands))

    def test_duplicate_json_key_is_rejected_before_execution(self):
        raw = b'{"schema_version":1,"schema_version":1}'
        result, code = self.helper.handle(raw)
        self.assertEqual((code, result["reason_code"]), (2, "malformed_request"))
        self.assertEqual(self.executor.commands, [])

    def test_malformed_non_utf8_nan_non_object_empty_and_oversized(self):
        cases = [b"\xff", b'{"x":NaN}', b"[]", b"", b"{" + b" " * helper_mod.MAX_REQUEST_BYTES]
        for raw in cases:
            with self.subTest(raw=raw[:20]):
                result, code = self.helper.handle(raw)
                self.assertEqual(code, 2)
                self.assertEqual(result["reason_code"], "malformed_request")

    def test_unknown_schema_policy_operation_and_fields_fail_closed(self):
        cases = []
        schema = valid_request()
        schema["schema_version"] = 2
        cases.append((schema, "unknown_schema"))
        policy = valid_request()
        policy["policy_version"] = 2
        cases.append((policy, "unknown_schema"))
        operation = valid_request([{"id": "x", "type": "shell.v1"}])
        cases.append((operation, "unknown_operation"))
        envelope = valid_request()
        envelope["command"] = "id"
        cases.append((envelope, "unknown_field"))
        nested = valid_request()
        nested["preconditions"]["cwd"] = "/tmp"
        cases.append((nested, "unknown_field"))
        for request, reason in cases:
            with self.subTest(reason=reason):
                result, code = self.handle(request)
                self.assertEqual(code, 2)
                self.assertEqual(result["reason_code"], reason)
        self.assertEqual(self.executor.commands, [])

    def test_all_arbitrary_execution_fields_are_rejected(self):
        forbidden = (
            "command", "executable", "argv", "shell", "script", "script_path",
            "source_path", "archive_path", "build_path", "destination_path", "cwd",
            "environment", "url", "download", "file_content", "module_name",
            "package_manager_flags",
        )
        for field in forbidden:
            operation = {"id": "packages", "type": "packages.ensure.v1", "packages": ["cmake"], field: "sentinel"}
            with self.subTest(field=field):
                result, code = self.handle(valid_request([operation]))
                self.assertEqual((code, result["reason_code"]), (2, "unknown_field"))
        self.assertEqual(self.executor.commands, [])

    def test_package_flags_versions_paths_duplicates_and_kernel_mismatch_rejected(self):
        packages = [
            ["--fix-broken"], ["cmake=3.22"], ["../cmake"], ["cmake", "cmake"],
            ["linux-headers-6.8.0-58-generic"], ["not-allowlisted"],
        ]
        for values in packages:
            request = valid_request([{"id": "p", "type": "packages.ensure.v1", "packages": values}])
            with self.subTest(values=values):
                result, code = self.handle(request)
                self.assertEqual((code, result["reason_code"]), (2, "invalid_package"))

    def test_unsupported_or_changed_os_and_kernel_rejected(self):
        for base, reason in [(('debian', '12'), "unsupported_os"), (('ubuntu', '22.04'), "unsupported_os")]:
            with self.subTest(base=base):
                self.system.base = base
                result, code = self.handle(valid_request())
                self.assertEqual((code, result["reason_code"]), (2, reason))
        self.system.base = ("ubuntu", "24.04")
        self.system.kernel = "6.8.0-58-generic"
        result, code = self.handle(valid_request())
        self.assertEqual((code, result["reason_code"]), (2, "precondition_changed"))

    def test_duplicate_operation_ids_and_operation_count_are_rejected(self):
        op = {"id": "same", "type": "packages.ensure.v1", "packages": ["cmake"]}
        result, code = self.handle(valid_request([op, dict(op)]))
        self.assertEqual((code, result["reason_code"]), (2, "invalid_argument"))
        many = [{"id": f"p{i}", "type": "packages.ensure.v1", "packages": ["cmake"]} for i in range(33)]
        result, code = self.handle(valid_request(many))
        self.assertEqual((code, result["reason_code"]), (2, "invalid_argument"))


class ModuleOperationTests(HelperHarness):
    @staticmethod
    def module_operation(operation_type="v4l.module.load.v1", **updates):
        operation = {
            "id": "camera",
            "type": operation_type,
            "device_number": 10,
            "label": "StudioCast Camera",
            "exclusive_caps": True,
        }
        if operation_type == "v4l.module.reload.v1":
            operation.update(module_unload_safe=True, device_not_busy=True)
        operation.update(updates)
        return operation

    def test_load_has_fixed_module_and_argument_array(self):
        result, code = self.handle(valid_request([self.module_operation()]))
        self.assertEqual((code, result["state"]), (0, "succeeded"))
        self.assertEqual(
            self.executor.commands[0][0],
            [
                "/usr/sbin/modprobe", "v4l2loopback", "devices=1", "video_nr=10",
                "card_label=StudioCast Camera", "exclusive_caps=1",
            ],
        )
        self.assertEqual(self.system.conflict_calls, [(10, False)])

    def test_device_conflict_blocks_without_command(self):
        self.system.conflict = True
        result, code = self.handle(valid_request([self.module_operation()]))
        self.assertEqual((code, result["reason_code"]), (3, "precondition_changed"))
        self.assertEqual(self.executor.commands, [])

    def test_reload_requires_explicit_safe_preconditions_and_live_idle_state(self):
        operation = self.module_operation("v4l.module.reload.v1", module_unload_safe=False)
        result, code = self.handle(valid_request([operation]))
        self.assertEqual((code, result["reason_code"]), (2, "precondition_changed"))
        self.system.busy = True
        operation["module_unload_safe"] = True
        result, code = self.handle(valid_request([operation]))
        self.assertEqual((code, result["reason_code"]), (3, "precondition_changed"))
        self.assertEqual(self.executor.commands, [])

    def test_reload_is_explicit_unload_then_load(self):
        result, code = self.handle(valid_request([self.module_operation("v4l.module.reload.v1")]))
        self.assertEqual(code, 0)
        commands = [command for command, _ in self.executor.commands]
        self.assertEqual(commands[0], ["/usr/sbin/modprobe", "-r", "v4l2loopback"])
        self.assertEqual(commands[1][0:3], ["/usr/sbin/modprobe", "v4l2loopback", "devices=1"])
        self.assertEqual(self.system.conflict_calls, [(10, True)])

    def test_device_caps_and_label_validation(self):
        bad_updates = [
            {"device_number": -1}, {"device_number": 64}, {"device_number": True},
            {"exclusive_caps": 1}, {"label": ""}, {"label": "a" * 65},
            {"label": "bad\nlabel"}, {"label": 'bad"label'}, {"label": "bad\\label"},
            {"label": "bad/label"},
        ]
        for updates in bad_updates:
            with self.subTest(updates=updates):
                result, code = self.handle(valid_request([self.module_operation(**updates)]))
                self.assertEqual(code, 2)
                self.assertEqual(result["reason_code"], "invalid_argument")

    def test_timeout_cancellation_and_failure_are_structured(self):
        cases = [
            (helper_mod.CommandResult(124, timed_out=True), "operation_timeout", 124),
            (helper_mod.CommandResult(130, cancelled=True), "operation_cancelled", 130),
            (helper_mod.CommandResult(9), "operation_failed", 9),
        ]
        for command_result, reason, exit_code in cases:
            with self.subTest(reason=reason):
                self.executor.results = [command_result]
                result, code = self.handle(valid_request([self.module_operation()]))
                self.assertEqual(code, 3)
                self.assertEqual(result["results"][0]["reason_code"], reason)
                self.assertEqual(result["results"][0]["exit_code"], exit_code)

    def test_prior_change_followed_by_failure_is_partial(self):
        first = self.module_operation()
        second = self.module_operation(device_number=11)
        second["id"] = "camera-two"
        self.executor.results = [helper_mod.CommandResult(0), helper_mod.CommandResult(5)]
        result, code = self.handle(valid_request([first, second]))
        self.assertEqual((code, result["state"], result["reason_code"]), (4, "partial_failure", "partial_failure"))
        self.assertEqual(len(result["results"]), 2)

    def test_production_executor_timeout_reaps_process_group(self):
        pid_file = self.root / "child.pid"
        code = (
            "import pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
            "time.sleep(30)"
        )
        started = time.monotonic()
        result = helper_mod.FixedCommandExecutor().run([sys.executable, "-c", code], timeout_seconds=0.2)
        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic() - started, 5)
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if Path(f"/proc/{child_pid}/stat").exists():
            state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
            self.assertEqual(state, "Z", "timed-out child process is still running")


class PersistenceTests(HelperHarness):
    def operation(self, operation_type="v4l.persistence.write.v1"):
        return {
            "id": "persist",
            "type": operation_type,
            "device_number": 10,
            "label": "StudioCast Camera (Main)+1",
            "exclusive_caps": True,
        }

    def paths(self):
        return (
            self.root / "modules-load.d/studiocast-v4l2loopback.conf",
            self.root / "modprobe.d/studiocast-v4l2loopback.conf",
        )

    def test_write_uses_only_namespaced_generated_files_and_hashes(self):
        legacy = self.root / "modules-load.d/v4l2loopback.conf"
        legacy.write_text("administrator owned\n")
        result, code = self.handle(valid_request([self.operation()]))
        self.assertEqual((code, result["state"]), (0, "succeeded"))
        modules, modprobe = self.paths()
        self.assertEqual(legacy.read_text(), "administrator owned\n")
        self.assertIn("schema=1 file=v4l_modules_load", modules.read_text())
        self.assertIn("card_label=StudioCast\\ Camera\\ (Main)+1", modprobe.read_text())
        self.assertNotIn('"', modprobe.read_text())
        self.assertEqual(stat.S_IMODE(modules.stat().st_mode), 0o644)
        returned = {item["id"]: item for item in result["results"][0]["files"]}
        for logical_id, path in zip(("v4l_modules_load", "v4l_modprobe"), self.paths()):
            self.assertEqual(returned[logical_id]["sha256"], "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(returned[logical_id]["path"].startswith("/etc/"))

    def test_symlink_nonregular_unmarked_and_wrong_owner_are_rejected(self):
        modules, _ = self.paths()
        target = self.root / "outside"
        target.write_text("untouched")
        modules.symlink_to(target)
        result, code = self.handle(valid_request([self.operation()]))
        self.assertEqual((code, result["results"][0]["reason_code"]), (3, "unsafe_path_state"))
        self.assertEqual(target.read_text(), "untouched")
        modules.unlink()
        modules.mkdir()
        result, _ = self.handle(valid_request([self.operation()]))
        self.assertEqual(result["results"][0]["reason_code"], "unsafe_path_state")
        modules.rmdir()
        modules.write_text("not StudioCast\n")
        modules.chmod(0o644)
        result, _ = self.handle(valid_request([self.operation()]))
        self.assertEqual(result["results"][0]["reason_code"], "unsafe_path_state")
        modules.write_text("# StudioCast system helper schema=1 file=v4l_modules_load\nv4l2loopback\n")
        modules.chmod(0o644)
        wrong_store = helper_mod.PersistenceStore(self.root, test_mode=True, required_uid=os.getuid() + 1, required_gid=os.getgid())
        wrong_helper = helper_mod.Helper(self.executor, self.system, wrong_store, test_mode=True)
        result, _ = wrong_helper.handle(encoded(valid_request([self.operation()])))
        self.assertEqual(result["results"][0]["reason_code"], "ownership_mismatch")

    def test_directory_symlink_is_rejected(self):
        (self.root / "modules-load.d").rmdir()
        (self.root / "modules-load.d").symlink_to(self.root / "modprobe.d", target_is_directory=True)
        result, code = self.handle(valid_request([self.operation()]))
        self.assertEqual((code, result["results"][0]["reason_code"]), (3, "unsafe_path_state"))

    def test_remove_requires_exact_hash_marker_owner_and_is_idempotent(self):
        write_result, _ = self.handle(valid_request([self.operation()]))
        hashes = {item["id"]: item["sha256"] for item in write_result["results"][0]["files"]}
        remove = {"id": "remove", "type": "v4l.persistence.remove.v1", "expected_hashes": hashes}
        result, code = self.handle(valid_request([remove]))
        self.assertEqual((code, result["state"]), (0, "succeeded"))
        self.assertTrue(all(not path.exists() for path in self.paths()))
        result, code = self.handle(valid_request([remove]))
        self.assertEqual(code, 0)
        self.assertFalse(result["results"][0]["changed"])

    def test_hash_mismatch_leaves_both_files_untouched(self):
        write_result, _ = self.handle(valid_request([self.operation()]))
        hashes = {item["id"]: item["sha256"] for item in write_result["results"][0]["files"]}
        hashes["v4l_modules_load"] = "sha256:" + "0" * 64
        before = [path.read_bytes() for path in self.paths()]
        remove = {"id": "remove", "type": "v4l.persistence.remove.v1", "expected_hashes": hashes}
        result, code = self.handle(valid_request([remove]))
        self.assertEqual((code, result["results"][0]["reason_code"]), (3, "hash_mismatch"))
        self.assertEqual(before, [path.read_bytes() for path in self.paths()])

    def test_test_mode_cannot_redirect_to_host_etc(self):
        with self.assertRaises(ValueError):
            helper_mod.PersistenceStore(Path("/etc"), test_mode=True)


class ClientContractTests(unittest.TestCase):
    def test_authorization_outcomes_have_stable_codes(self):
        for status, reason in client_mod.AUTHORIZATION_REASONS.items():
            self.assertEqual(client_mod.authorization_record(status), {"status": status, "reason_code": reason})
        with self.assertRaises(ValueError):
            client_mod.authorization_record("password_prompt")

    def test_helper_trust_missing_symlink_permissions_owner_version_and_success(self):
        with tempfile.TemporaryDirectory(prefix="studiocast-helper-trust-") as directory:
            root = Path(directory)
            path = root / "helper"
            version = lambda _path: '{"helper_version":"1.0.0","protocol_versions":[1]}'
            result = client_mod.probe_helper(path, version_probe=version, required_uid=os.getuid(), test_mode=True)
            self.assertEqual(result.reason_code, "privileged_helper_missing")
            target = root / "target"
            target.write_text("x")
            path.symlink_to(target)
            result = client_mod.probe_helper(path, version_probe=version, required_uid=os.getuid(), test_mode=True)
            self.assertEqual(result.reason_code, "privileged_helper_untrusted")
            path.unlink()
            path.write_text("helper")
            path.chmod(0o777)
            result = client_mod.probe_helper(path, version_probe=version, required_uid=os.getuid(), test_mode=True)
            self.assertEqual(result.reason_code, "privileged_helper_untrusted")
            path.chmod(0o755)
            result = client_mod.probe_helper(path, version_probe=version, required_uid=os.getuid() + 1, test_mode=True)
            self.assertEqual(result.reason_code, "privileged_helper_untrusted")
            result = client_mod.probe_helper(path, version_probe=lambda _: '{}', required_uid=os.getuid(), test_mode=True)
            self.assertEqual(result.reason_code, "privileged_helper_incompatible")
            result = client_mod.probe_helper(path, version_probe=version, required_uid=os.getuid(), test_mode=True)
            self.assertTrue(result.trusted)


class PackagingContractTests(unittest.TestCase):
    def test_layout_and_polkit_are_fixed_and_root_owned(self):
        layout_path = REPO_ROOT / "packaging/system-helper/package-layout.json"
        layout = json.loads(layout_path.read_text())
        files = {item["destination"]: item for item in layout["files"]}
        self.assertEqual(
            layout["authorized_entrypoint"],
            {
                "path": "/usr/libexec/studiocast/studiocast-system-helper",
                "arguments": ["--json-stdin"],
            },
        )
        self.assertEqual(
            set(files),
            {
                "/usr/libexec/studiocast/studiocast-system-helper",
                "/usr/share/polkit-1/actions/com.studiocast.Installer1.policy",
            },
        )
        self.assertEqual((files["/usr/libexec/studiocast/studiocast-system-helper"]["uid"], files["/usr/libexec/studiocast/studiocast-system-helper"]["gid"], files["/usr/libexec/studiocast/studiocast-system-helper"]["mode"]), (0, 0, "0755"))
        policy = (REPO_ROOT / "packaging/system-helper/com.studiocast.Installer1.policy").read_text()
        self.assertIn("/usr/libexec/studiocast/studiocast-system-helper", policy)
        self.assertIn("auth_admin_keep", policy)
        self.assertNotIn("sudo", policy)
        self.assertEqual(
            HELPER_SOURCE.read_text().splitlines()[0],
            "#!/usr/bin/python3 -I",
            "privileged Python must ignore caller-controlled Python paths",
        )

    def test_selected_source_sentinel_is_never_executed(self):
        with tempfile.TemporaryDirectory(prefix="studiocast-selected-source-") as directory:
            root = Path(directory)
            sentinel = root / "executed"
            setup = root / "setup.sh"
            setup.write_text(f"#!/bin/sh\ntouch {sentinel}\n")
            setup.chmod(0o755)
            fake_root = root / "etc"
            (fake_root / "modules-load.d").mkdir(parents=True)
            (fake_root / "modprobe.d").mkdir()
            executor = FakeExecutor()
            system = FakeSystem()
            store = helper_mod.PersistenceStore(fake_root, test_mode=True, required_uid=os.getuid(), required_gid=os.getgid())
            instance = helper_mod.Helper(executor, system, store, test_mode=True)
            request = valid_request()
            request["operations"][0]["source_path"] = str(setup)
            result, code = instance.handle(encoded(request))
            self.assertEqual((code, result["reason_code"]), (2, "unknown_field"))
            self.assertFalse(sentinel.exists())
            self.assertEqual(executor.commands, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
