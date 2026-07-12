#!/usr/bin/env python3
"""Hermetic contract tests for the StudioCast installer backend."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SOURCE = Path(os.environ.get("STUDIOCAST_SOURCE_DIR", Path(__file__).resolve().parents[1]))
BACKEND = SOURCE / "installer/backend/studiocast-installer-backend"
DEFAULT_PACKS = [
    "fastenhancer_s_vd_v1", "fastenhancer_m_vd_v1", "modnet-webnn-256-fp32",
    "yunet_opencv_zoo_2023mar_fp32", "dlib_68_ibug_300w",
    "gaze_correction_cam_flx_v0_1_1", "fastdvdnet_sigma15",
]


def base_facts(**installation):
    install = {"classification": "absent", "active_version": None,
               "previous_version": None, "target_version": "0.2.9",
               "version_relation": "not_installed", "desired_configuration": None,
               "reason_codes": ["install.manifest.absent"]}
    install.update(installation)
    return {
        "schema_version": 1, "facts_version": "installer-facts/v1", "os": {
            "id": "ubuntu", "version_id": "24.04", "base_id": "ubuntu",
            "base_version_id": "24.04", "architecture": "x86_64", "kernel_release": "fixture",
            "supported": True, "reason_codes": ["os.supported.ubuntu_noble"]},
        "installation": install,
        "paths": {"free_bytes": 20_000_000_000,
                  "required_bytes": {"download": 1, "build": 1, "staging": 1}},
        "toolchain": {"tools": {}, "missing_build_dependencies": [], "reason_codes": []},
        "runtime": {"onnxruntime": {"present": True, "compatible": True,
                    "providers": {"cpu": True, "cuda": True}, "probe": "usable"}},
        "systemd_user": {"systemctl_present": True, "manager_usable": True,
                         "reason_codes": ["service.user_manager.usable"]},
        "v4l2": {"package_state": "installed", "module_available": True,
                  "module_loaded": True, "device_present": True, "device_number": 10,
                  "device_number_conflict": False, "persistence_state": "configured",
                  "kernel_headers_usable": True, "dkms_viable": True,
                  "secure_boot": "disabled", "reason_codes": ["v4l.device.present"]},
        "privileged_helper": {"present": False, "trusted": False, "compatible": False,
                              "reason_codes": ["privileged_helper_missing"]},
        "gpus": {"devices": [], "nvidia": {"driver_usable": False, "cuda_usable": False},
                 "vulkan": {"loader_present": False, "physical_devices": [],
                            "compute_device_usable": False}},
        "maxine": {"components": {}, "effects": {}, "reason_codes": []},
        "effects": {"capabilities": {
            "virtual_background": {"maxine": "unavailable", "cuda": "production_usable",
                                   "vulkan": "production_usable", "cpu": "unavailable"}}},
        "models": {"packs": {}, "default_pack_ids": DEFAULT_PACKS},
        "cache": {"release_artifacts": {}, "model_artifacts": {}},
        "connectivity": {"release_source": {"state": "online", "reason_codes": []},
                         "model_source": {"state": "online", "reason_codes": []}},
        "reason_codes": [],
    }


class Sandbox:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="studiocast-installer-core-")
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env.update({"HOME": str(self.root / "home"), "XDG_DATA_HOME": str(self.root / "data"),
                         "XDG_CONFIG_HOME": str(self.root / "config"), "XDG_STATE_HOME": str(self.root / "state"),
                         "XDG_CACHE_HOME": str(self.root / "cache"),
                         "STUDIOCAST_INSTALLER_OFFLINE": "1"})
        self.facts = self.root / "facts.json"
        self.write_facts(base_facts())

    def close(self): self.temp.cleanup()

    def write_facts(self, facts): self.facts.write_text(json.dumps(facts), encoding="utf-8")

    def run(self, *args, check=True, env=None):
        result = subprocess.run([str(BACKEND), *args], env=env or self.env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and result.returncode:
            raise AssertionError(f"backend failed ({result.returncode}):\nstdout={result.stdout}\nstderr={result.stderr}")
        return result

    def json(self, *args, **kwargs): return json.loads(self.run(*args, **kwargs).stdout)


class InstallerBackendCoreTests(unittest.TestCase):
    def setUp(self): self.s = Sandbox()
    def tearDown(self): self.s.close()

    def recommendation(self, facts=None, *extra):
        if facts is not None: self.s.write_facts(facts)
        return self.s.json("recommend", "--facts", str(self.s.facts), "--intent", "install", *extra)

    def plan(self, *extra):
        return self.s.json("plan", "install", "--facts", str(self.s.facts),
                           "--source-dir", str(SOURCE), "--no-v4l2loopback",
                           "--no-service", "--no-models", "--allow-unsupported", *extra)

    def test_fixture_analysis_is_exact_no_host_probe(self):
        marker = self.s.root / "marker"
        fake = self.s.root / "bin/systemctl"; fake.parent.mkdir();
        fake.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n", encoding="utf-8"); fake.chmod(0o755)
        env = self.s.env | {"PATH": str(fake.parent) + ":" + self.s.env.get("PATH", "")}
        first = self.s.json("analyze", "--input-fixture", str(self.s.facts), env=env)
        second = self.s.json("analyze", "--input-fixture", str(self.s.facts), env=env)
        self.assertFalse(marker.exists())
        self.assertEqual(first, second)
        self.assertTrue(first["host_fingerprint"].startswith("sha256:"))

    def test_routes_cover_install_update_modify_newer_repair_tombstone_unsupported_offline(self):
        cases = [
            (base_facts(), ("recommended", "install")),
            (base_facts(classification="healthy", active_version="0.2.8", version_relation="upgrade"), ("recommended", "update")),
            (base_facts(classification="healthy", active_version="0.2.9", version_relation="same"), ("modify", "modify")),
            (base_facts(classification="healthy", active_version="0.3.0", version_relation="installed_newer"), ("current_newer", "keep")),
            (base_facts(classification="partial"), ("repair", "repair")),
            (base_facts(classification="tombstone"), ("repair", "reconstruct")),
        ]
        for facts, expected in cases:
            with self.subTest(expected=expected):
                rec = self.recommendation(facts)
                self.assertEqual((rec["route"], rec["primary_action"]), expected)
        facts = base_facts(); facts["os"]["supported"] = False
        self.assertEqual(self.recommendation(facts)["route"], "unsupported")
        facts = base_facts(); facts["connectivity"]["release_source"]["state"] = "offline"
        self.assertEqual(self.recommendation(facts)["route"], "offline")

    def test_recommendation_is_deterministic_models_exact_and_vulkan_not_promoted(self):
        one = self.recommendation(); two = self.recommendation()
        self.assertEqual(one, two)
        self.assertEqual(one["selections"]["model_pack_ids"], DEFAULT_PACKS)
        self.assertEqual(one["selections"]["effects"]["virtual_background"]["selected"], "cuda")
        facts = base_facts(); facts["effects"]["capabilities"]["virtual_background"]["cuda"] = "unavailable"
        self.assertEqual(self.recommendation(facts)["selections"]["effects"]["virtual_background"]["selected"], "unavailable")
        facts["maxine"] = {"components": {"video_fx": {"usable": True}}, "effects": {}, "reason_codes": []}
        facts["effects"]["capabilities"]["virtual_background"]["maxine"] = "production_usable"
        self.assertEqual(self.recommendation(facts)["selections"]["effects"]["virtual_background"]["selected"], "maxine")
        self.assertEqual(self.recommendation(facts)["selections"]["model_pack_ids"], DEFAULT_PACKS)

    def test_plan_has_exact_explicit_flags_digest_and_v4l_only_typed_ops(self):
        plan = self.plan("--no-open-backends", "--no-open-vulkan")
        configure = next(x for x in plan["operations"] if x["id"] == "build.configure")
        self.assertEqual(configure["inputs"]["cmake"], {
            "STUDIOCAST_ENABLE_OPEN_AUDIO": False, "STUDIOCAST_ENABLE_OPEN_CUDA": False,
            "STUDIOCAST_ENABLE_OPEN_VULKAN": False})
        self.assertTrue(plan["plan_digest"].startswith("sha256:"))
        facts = base_facts(); facts["v4l2"].update(module_available=False, module_loaded=False,
                                                  device_present=False, persistence_state="absent")
        facts["privileged_helper"].update(present=True, trusted=True, compatible=True)
        self.s.write_facts(facts)
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                           "--skip-deps", "--v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
        ids = [x["id"] for x in plan["operations"]]
        self.assertIn("v4l.module.load", ids); self.assertIn("v4l.persistence.write", ids)
        self.assertNotIn("scripts/setup.sh", json.dumps(plan))
        self.assertTrue(all(x["privilege"] == "trusted_helper" for x in plan["operations"] if x["id"].startswith("v4l.")))
        for op in (item for item in plan["operations"] if item["id"] in {"v4l.module.load", "v4l.persistence.write"}):
            self.assertEqual(set(op["inputs"]), {"operation_type", "device_number", "label", "exclusive_caps"})

    def test_apply_rejects_tamper_and_exact_dry_run_executes(self):
        plan = self.plan()
        path = self.s.root / "plan.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], "--facts", str(self.s.facts), "--dry-run")
        value = json.loads(result.stdout[result.stdout.find("{"):])
        self.assertEqual(value["reviewed_operation_ids"], value["executed_operation_ids"])
        plan = self.plan(); plan["operations"][1]["id"] = "tampered"
        path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], check=False)
        self.assertNotEqual(result.returncode, 0); self.assertIn("plan.digest_mismatch", result.stderr)
        plan = self.plan(); plan["plan_id"] = "../../outside"
        path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], check=False)
        self.assertEqual(result.returncode, 2); self.assertIn("plan.invalid_id", result.stderr)

    def test_manifest_v1_migration_does_not_trust_paths(self):
        path = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/install-manifest.json"; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema_version": 1, "state": "installed", "installed_version": "0.2.8",
                                    "source_path": "/", "build_path": "/tmp/evil"}), encoding="utf-8")
        value = self.s.json("manifest", "--migrate-v1")
        self.assertEqual(value["schema_version"], 2)
        self.assertEqual(value["ownership"]["application_files"], [])
        self.assertEqual(value["source"]["legacy_hints"]["source_path"], "/")
        self.assertEqual(json.loads(path.read_text())["schema_version"], 2)

    def test_pack_catalog_is_exact_seven_ids_eight_hashed_artifacts(self):
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                           "--no-v4l2loopback", "--no-service", "--models", "--allow-unsupported")
        self.assertEqual([x["artifact_id"] for x in plan["downloads"]], DEFAULT_PACKS)
        verify = [x for x in plan["operations"] if x["kind"] == "model_verify"]
        artifacts = [a for op in verify for a in op["inputs"]["artifacts"]]
        self.assertEqual(len(artifacts), 8)
        self.assertTrue(all(a["sha256"].startswith("sha256:") for a in artifacts))
        self.assertEqual([a["path"] for a in next(x for x in verify if x["inputs"]["pack_id"] == "gaze_correction_cam_flx_v0_1_1")["inputs"]["artifacts"]],
                         ["gaze_flx_left.onnx", "gaze_flx_right.onnx"])

    def test_packaged_style_uninstall_is_self_contained_and_preserves_data(self):
        data = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"
        payload = data / "payloads/0.2.9/bin"; payload.mkdir(parents=True)
        current = data / "current"; current.symlink_to(data / "payloads/0.2.9")
        models = data / "models/custom"; models.mkdir(parents=True); (models / "mine").write_text("keep")
        config = Path(self.s.env["XDG_CONFIG_HOME"]) / "studiocast"; config.mkdir(parents=True); (config / "daemon.conf").write_text("keep")
        plan = self.s.json("plan", "uninstall", "--facts", str(self.s.facts), "--preserve-user-data")
        path = self.s.root / "uninstall.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"], "--token", plan["approval_token"])
        self.assertEqual(json.loads(result.stdout)["state"], "committed")
        self.assertFalse((data / "payloads").exists()); self.assertTrue((models / "mine").exists()); self.assertTrue((config / "daemon.conf").exists())
        self.assertEqual(json.loads((data / "install-manifest.json").read_text())["transaction"]["state"], "uninstalled")

    def test_uninstall_removes_only_hash_bound_v4l_persistence(self):
        facts = base_facts()
        facts["privileged_helper"].update(present=True, trusted=True, compatible=True)
        facts["v4l2"]["owned_configuration"] = [
            {"id": "v4l_modules_load", "path": "/etc/modules-load.d/studiocast-v4l2loopback.conf",
             "sha256": "sha256:" + "1" * 64},
            {"id": "v4l_modprobe", "path": "/etc/modprobe.d/studiocast-v4l2loopback.conf",
             "sha256": "sha256:" + "2" * 64},
        ]
        self.s.write_facts(facts)
        plan = self.s.json("plan", "uninstall", "--facts", str(self.s.facts), "--preserve-user-data")
        remove = next(item for item in plan["operations"] if item["id"] == "v4l.persistence.remove")
        self.assertEqual(remove["failure_policy"], "degrade")
        self.assertEqual(remove["inputs"], {"operation_type": "v4l.persistence.remove.v1",
            "expected_hashes": {"v4l_modules_load": "sha256:" + "1" * 64,
                                "v4l_modprobe": "sha256:" + "2" * 64}})
        path = self.s.root / "uninstall-v4l.json"; path.write_text(json.dumps(plan))
        helper = self.s.root / "helper"; pkexec = self.s.root / "pkexec"
        for executable in (helper, pkexec):
            executable.write_text("#!/bin/sh\nexit 99\n"); executable.chmod(0o755)
        env = self.s.env | {"STUDIOCAST_INSTALLER_TEST_MODE": "1",
                            "STUDIOCAST_INSTALLER_TEST_HELPER": str(helper),
                            "STUDIOCAST_INSTALLER_TEST_PKEXEC": str(pkexec)}
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], "--facts", str(self.s.facts), "--dry-run", env=env)
        value = json.loads(result.stdout[result.stdout.find("{"):])
        self.assertEqual(value["reviewed_operation_ids"], value["executed_operation_ids"])

    def test_host_analyzer_consumes_runtime_diagnostics_and_real_vulkan_compute_evidence(self):
        current_bin = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/current/bin"
        current_bin.mkdir(parents=True)
        diagnostics = {"engines": {
            "maxine": {"ok": True, "available_effects": ["auto_frame"], "components": {},
                       "vfx": {"root_found": True, "library_loadable": True, "ok": True},
                       "ar": {"root_found": False, "library_loadable": False, "ok": False},
                       "afx": {"root_found": False, "library_loadable": False, "ok": False}},
            "open_cuda": {"onnxruntime_version": "1.20.0",
                          "onnxruntime_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                          "onnxruntime_cuda_provider_present": True,
                          "onnxruntime_cpu_provider_present": True, "cuda_context_available": True,
                          "available_effects": ["virtual_background.blur"]},
            "open_audio": {}, "open_vulkan": {"available_effects": ["auto_frame"]}}}
        ctl = current_bin / "studiocastctl"
        ctl.write_text("#!/bin/sh\nprintf '%s\\n' " + repr(json.dumps(diagnostics)) + "\n")
        ctl.chmod(0o755)
        fake = self.s.root / "probe-bin"; fake.mkdir()
        (fake / "vulkaninfo").write_text("""#!/bin/sh
if [ "${1:-}" = --summary ]; then
cat <<'EOF'
GPU0:
    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName = Test GPU
EOF
else
printf 'GPU id = 0 (Test GPU)\\nqueueFlags = VK_QUEUE_COMPUTE_BIT\\n'
fi
""")
        (fake / "nvidia-smi").write_text("""#!/bin/sh
case "${1:-}" in --query-gpu=*) printf '0, GPU-test, Test GPU, 550.1, 8.6\\n';; *) printf 'CUDA Version: 12.4\\n';; esac
""")
        for name in ("vulkaninfo", "nvidia-smi"):
            (fake / name).chmod(0o755)
        env = self.s.env | {"PATH": str(fake) + ":/usr/bin:/bin"}
        facts = self.s.json("analyze", "--target-version", "0.2.9", env=env)
        self.assertTrue(facts["gpus"]["vulkan"]["compute_device_usable"])
        self.assertEqual(facts["gpus"]["vulkan"]["reason_codes"], ["vulkan_compute_device_usable"])
        self.assertTrue(facts["gpus"]["nvidia"]["cuda_usable"])
        self.assertEqual(facts["runtime"]["onnxruntime"]["probe"], "daemon_runtime_diagnostics")
        self.assertTrue(facts["runtime"]["onnxruntime"]["providers"]["cuda"])
        self.assertEqual(facts["effects"]["capabilities"]["auto_frame"]["maxine"], "production_usable")
        self.assertEqual(facts["effects"]["capabilities"]["auto_frame"]["vulkan"],
                         "usable_with_degraded_behavior")

    def test_unsupported_override_is_advanced_only_and_rewires_removed_privilege(self):
        facts = base_facts(); facts["os"]["supported"] = False
        facts["os"]["reason_codes"] = ["os.unsupported.distribution"]
        facts["v4l2"].update(module_available=False, module_loaded=False, device_present=False,
                             persistence_state="absent")
        facts["privileged_helper"].update(present=True, trusted=True, compatible=True)
        self.s.write_facts(facts)
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                           "--v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
        self.assertFalse(any(item["privilege"] == "trusted_helper" for item in plan["operations"]))
        previous = None
        for op in plan["operations"]:
            self.assertEqual(op["depends_on"], [] if previous is None else [previous])
            previous = op["id"]

        # Without an Advanced source selection the same override remains blocked.
        blocked = self.s.json("plan", "install", "--facts", str(self.s.facts),
                              "--no-v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
        self.assertIn("os.unsupported.override_requires_advanced", [item["code"] for item in blocked["blockers"]])

    def test_repair_healthy_core_is_minimal_and_reuses_prior_configuration(self):
        facts = base_facts(classification="healthy", active_version="0.2.9", target_version="0.2.9",
                           version_relation="same", links_healthy=True,
                           desired_configuration={"build_type": "Debug", "features": {"open_cuda": False, "open_audio": True, "open_vulkan": False},
                                                  "service": {"desired": "disabled"},
                                                  "v4l": {"desired": False, "required_for_success": False},
                                                  "model_pack_ids": ["fastenhancer_s_vd_v1"], "preserve_settings": True})
        facts["service"] = {"actual": "inactive"}
        facts["models"]["packs"] = {"fastenhancer_s_vd_v1": {"integrity": "verified"}}
        self.s.write_facts(facts)
        plan = self.s.json("plan", "repair", "--facts", str(self.s.facts), "--source-dir", str(SOURCE), "--allow-unsupported")
        ids = [op["id"] for op in plan["operations"]]
        self.assertEqual(ids, ["preflight.validate", "manifest.commit"])
        self.assertTrue(plan["core_reuse"])
        self.assertEqual(plan["desired_state"]["build_type"], "Debug")
        self.assertNotIn("build.configure", ids)

    def _fake_cmake(self, fail=False):
        bindir = self.s.root / "fake-bin"; bindir.mkdir(exist_ok=True)
        cmake = bindir / "cmake"
        cmake.write_text("#!/bin/sh\n" + ("exit 41\n" if fail else "if [ \"$1\" = --build ]; then\n d=$2\n mkdir -p \"$d\"\n for n in studiocast studiocastd studiocastctl studiocast-probe studiocast-open studiocast-maxine studiocast-video studiocast-audio; do printf '#!/bin/sh\\n[ \"$1\" = --version ] && echo StudioCast-0.2.9\\nexit 0\\n' >\"$d/$n\"; chmod 755 \"$d/$n\"; done\nfi\nexit 0\n"), encoding="utf-8")
        cmake.chmod(0o755)
        return self.s.env | {"PATH": str(bindir) + ":/usr/bin:/bin"}

    def _apply_clean(self, preserve):
        args = ["plan", "clean-install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                "--no-v4l2loopback", "--no-service", "--no-models", "--allow-unsupported",
                "--preserve-user-data" if preserve else "--remove-user-data"]
        plan = self.s.json(*args)
        path = self.s.root / "clean.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        return self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                          "--token", plan["approval_token"], check=False, env=self._fake_cmake()), plan

    def test_clean_install_preserves_only_settings_or_performs_literal_wipe(self):
        config = Path(self.s.env["XDG_CONFIG_HOME"]) / "studiocast"; config.mkdir(parents=True); (config / "daemon.conf").write_text("chosen")
        data = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"; (data / "models/custom").mkdir(parents=True); (data / "models/custom/mine").write_text("remove")
        state = Path(self.s.env["XDG_STATE_HOME"]) / "studiocast"; state.mkdir(parents=True); (state / "log").write_text("remove")
        result, plan = self._apply_clean(True)
        self.assertEqual(result.returncode, 3)  # core committed; uncached optional models degraded
        self.assertEqual((config / "daemon.conf").read_text(), "chosen")
        self.assertFalse((data / "models/custom/mine").exists()); self.assertFalse((state / "log").exists())
        self.assertFalse(Path(plan["preservation"]["snapshot_path"]).exists())

        # A literal full wipe does not recreate or restore the user's settings.
        (config / "daemon.conf").write_text("remove")
        (data / "models/custom").mkdir(parents=True, exist_ok=True); (data / "models/custom/mine").write_text("remove")
        self.s.write_facts(self.s.json("analyze", "--target-version", "0.2.9"))
        result, _ = self._apply_clean(False)
        self.assertEqual(result.returncode, 3)
        self.assertFalse(config.exists()); self.assertFalse((data / "models/custom/mine").exists())

    def test_clean_failure_restores_settings_and_preserves_non_build_caches(self):
        config = Path(self.s.env["XDG_CONFIG_HOME"]) / "studiocast"
        config.mkdir(parents=True); (config / "daemon.conf").write_text("preserved")
        cache = Path(self.s.env["XDG_CACHE_HOME"]) / "studiocast"
        for relative in ("release/artifacts/source.tar", "models/custom/model.onnx", "builds/old/junk", "sources/old/junk"):
            path = cache / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
        # A fatal configure failure exercises compensation after settings and
        # old build caches have already been removed.
        plan = self.s.json("plan", "clean-install", "--facts", str(self.s.facts),
                           "--source-dir", str(SOURCE), "--no-v4l2loopback", "--no-service",
                           "--no-models", "--allow-unsupported", "--preserve-user-data")
        path = self.s.root / "clean-fail.json"; path.write_text(json.dumps(plan))
        failed = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], check=False, env=self._fake_cmake(fail=True))
        self.assertEqual(failed.returncode, 2)
        self.assertEqual((config / "daemon.conf").read_text(), "preserved")
        self.assertTrue((cache / "release/artifacts/source.tar").is_file())
        self.assertTrue((cache / "models/custom/model.onnx").is_file())
        self.assertFalse((cache / "sources/old/junk").exists())
        manifest = json.loads((Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/install-manifest.json").read_text())
        self.assertIn("settings.restore.compensation", [item["operation_id"] for item in manifest["journal"]])

    def test_versioned_payload_survives_cache_removal_and_core_failure_rolls_back(self):
        plan = self.plan()
        path = self.s.root / "install.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], env=self._fake_cmake())
        self.assertEqual(json.loads(result.stdout)["state"], "committed")
        current = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/current"
        active_before = current.resolve()
        cache = Path(self.s.env["XDG_CACHE_HOME"]) / "studiocast"
        import shutil
        shutil.rmtree(cache)
        self.assertEqual(subprocess.run([str(current / "bin/studiocast")]).returncode, 0)

        self.s.write_facts(self.s.json("analyze", "--target-version", "0.2.9"))
        update = self.s.json("plan", "update", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                             "--no-v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
        path.write_text(json.dumps(update), encoding="utf-8")
        failed = self.s.run("apply-plan", "--plan", str(path), "--digest", update["plan_digest"],
                            "--token", update["approval_token"], check=False, env=self._fake_cmake(fail=True))
        self.assertEqual(failed.returncode, 2)
        self.assertEqual(current.resolve(), active_before)
        manifest = json.loads((Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/install-manifest.json").read_text())
        self.assertEqual(manifest["payloads"]["active"]["path"], str(active_before))
        self.assertEqual(manifest["transaction"]["state"], "failed")

    def test_payload_validation_runs_version_probe_before_activation(self):
        plan = self.plan()
        path = self.s.root / "version-plan.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        env = self._fake_cmake()
        cmake = self.s.root / "fake-bin/cmake"
        cmake.write_text(cmake.read_text().replace("StudioCast-0.2.9", "StudioCast-9.9.9"), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], check=False, env=env)
        self.assertEqual(result.returncode, 2)
        value = json.loads(result.stdout)
        self.assertEqual(value["error"]["code"], "payload.version_mismatch")
        self.assertFalse((Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/current").exists())

    def test_cancellation_terminates_process_group_and_journals_cancelled_state(self):
        plan = self.plan()
        path = self.s.root / "cancel-plan.json"; path.write_text(json.dumps(plan))
        fake = self.s.root / "cancel-bin"; fake.mkdir()
        child_pid = self.s.root / "child.pid"
        cmake = fake / "cmake"
        cmake.write_text(f"""#!/bin/sh
if [ "${{1:-}}" = -S ]; then
  sleep 60 &
  printf '%s' "$!" >'{child_pid}'
  wait
fi
exit 0
"""); cmake.chmod(0o755)
        env = self.s.env | {"PATH": str(fake) + ":/usr/bin:/bin"}
        process = subprocess.Popen([str(BACKEND), "apply-plan", "--plan", str(path),
                                    "--digest", plan["plan_digest"], "--token", plan["approval_token"],
                                    "--facts", str(self.s.facts)], env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(child_pid.exists(), "fake build child did not start")
        pid = int(child_pid.read_text())
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 130, stderr)
        self.assertEqual(json.loads(stdout)["state"], "cancelled")
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            self.fail("cancelled build child was not reaped")
        manifest = json.loads((Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/install-manifest.json").read_text())
        self.assertEqual(manifest["transaction"]["state"], "cancelled")
        cancelled = next(item for item in manifest["journal"] if item["operation_id"] == "build.configure")
        self.assertEqual(cancelled["state"], "cancelled")

    def test_fake_polkit_transport_envelope_and_structured_failures(self):
        facts = base_facts(); facts["v4l2"].update(module_available=True, module_loaded=False,
                                                  device_present=False, persistence_state="configured")
        facts["privileged_helper"].update(present=True, trusted=True, compatible=True)
        self.s.write_facts(facts)
        adapter = self.s.root / "auth/pkexec"; helper = self.s.root / "auth/helper"; adapter.parent.mkdir()
        helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8"); helper.chmod(0o755)
        captured = self.s.root / "request.json"
        base_env = self._fake_cmake() | {"STUDIOCAST_INSTALLER_TEST_MODE": "1",
            "STUDIOCAST_INSTALLER_TEST_HELPER": str(helper), "STUDIOCAST_INSTALLER_TEST_PKEXEC": str(adapter),
            "STUDIOCAST_INSTALLER_TEST_REQUEST": str(captured)}

        def invoke(script, timeout=None):
            # Each authorization outcome starts from the same absent fixture;
            # a preceding success must not make the next plan stale.
            import shutil
            data_root = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"
            if data_root.exists(): shutil.rmtree(data_root)
            bin_root = Path(self.s.env["HOME"]) / ".local/bin"
            if bin_root.exists():
                for entry in bin_root.iterdir():
                    if entry.is_symlink(): entry.unlink()
            adapter.write_text("#!/bin/sh\n" + script, encoding="utf-8"); adapter.chmod(0o755)
            plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                               "--skip-deps", "--v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
            path = self.s.root / "helper-plan.json"; path.write_text(json.dumps(plan), encoding="utf-8")
            env = dict(base_env)
            if timeout: env["STUDIOCAST_INSTALLER_TEST_AUTH_TIMEOUT"] = timeout
            result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                                "--token", plan["approval_token"], check=False, env=env)
            return result, json.loads(result.stdout) if result.stdout.strip().startswith("{") else None

        success_script = f"""cat >'{captured}'
python3 -c 'import json; p={json.dumps(str(captured))}; r=json.load(open(p)); o=r[\"operations\"][0]; print(json.dumps({{\"schema_version\":1,\"request_id\":r[\"request_id\"],\"transaction_id\":r[\"transaction_id\"],\"plan_digest\":r[\"plan_digest\"],\"state\":\"succeeded\",\"results\":[{{\"id\":o[\"id\"],\"type\":o[\"type\"],\"status\":\"succeeded\",\"files\":[]}}]}}))'
"""
        success, value = invoke(success_script)
        self.assertEqual(success.returncode, 0); self.assertEqual(value["state"], "committed")
        request = json.loads(captured.read_text())
        self.assertEqual(request["policy_version"], 1)
        self.assertEqual(request["preconditions"], {"base_os": "ubuntu", "base_release": "24.04", "kernel_release": "fixture"})
        self.assertEqual(request["operations"][0]["type"], "v4l.module.load.v1")
        self.assertIn("device_number", request["operations"][0]); self.assertNotIn("arguments", request["operations"][0])

        for exit_code, reason in ((126, "authorization_denied"), (127, "authorization_cancelled")):
            result, value = invoke(f"exit {exit_code}\n")
            self.assertEqual(result.returncode, 2); self.assertEqual(value["error"]["code"], reason)
        result, value = invoke("printf 'not-json\\n'\n", None)
        self.assertEqual(result.returncode, 2); self.assertEqual(value["error"]["code"], "privilege.malformed_result")
        result, value = invoke("printf '{\"schema_version\":1,\"state\":\"succeeded\",\"results\":[]}\\n'\n")
        self.assertEqual(result.returncode, 2); self.assertEqual(value["error"]["code"], "privilege.result_binding_mismatch")
        result, value = invoke("sleep 1\n", "0.05")
        self.assertEqual(result.returncode, 2); self.assertEqual(value["error"]["code"], "authorization_timeout")

    def test_persistence_helper_hashes_are_recorded_as_owned_manifest_state(self):
        facts = base_facts()
        facts["v4l2"].update(module_available=True, module_loaded=True, device_present=True,
                             persistence_state="absent")
        facts["privileged_helper"].update(present=True, trusted=True, compatible=True)
        self.s.write_facts(facts)
        adapter = self.s.root / "persist-pkexec"
        helper = self.s.root / "persist-helper"
        helper.write_text("#!/bin/sh\nexit 0\n"); helper.chmod(0o755)
        adapter.write_text("""#!/usr/bin/python3
import json, sys
r = json.load(sys.stdin)
o = r["operations"][0]
files = [
 {"id":"v4l_modules_load","path":"/etc/modules-load.d/studiocast-v4l2loopback.conf","sha256":"sha256:" + "1"*64},
 {"id":"v4l_modprobe","path":"/etc/modprobe.d/studiocast-v4l2loopback.conf","sha256":"sha256:" + "2"*64},
]
print(json.dumps({"schema_version":1,"request_id":r["request_id"],"transaction_id":r["transaction_id"],
 "plan_digest":r["plan_digest"],"state":"succeeded","results":[{"id":o["id"],"type":o["type"],
 "status":"succeeded","files":files}]}))
"""); adapter.chmod(0o755)
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                           "--v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
        path = self.s.root / "persist-plan.json"; path.write_text(json.dumps(plan))
        env = self._fake_cmake() | {"STUDIOCAST_INSTALLER_TEST_MODE": "1",
            "STUDIOCAST_INSTALLER_TEST_HELPER": str(helper),
            "STUDIOCAST_INSTALLER_TEST_PKEXEC": str(adapter)}
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], "--facts", str(self.s.facts), env=env)
        self.assertEqual(json.loads(result.stdout)["state"], "committed")
        manifest = json.loads((Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/install-manifest.json").read_text())
        self.assertEqual(manifest["v4l"]["actual"]["persistence_state"], "configured")
        self.assertEqual(len(manifest["v4l"]["owned_configuration"]), 2)
        self.assertEqual(len(manifest["ownership"]["system_configuration"]), 2)

    def test_unknown_and_corrupt_manifest_fail_closed(self):
        path = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/install-manifest.json"; path.parent.mkdir(parents=True)
        path.write_text('{"schema_version":', encoding="utf-8")
        status = self.s.json("status", "--target-version", "0.2.9")
        self.assertEqual(status["classification"], "unhealthy")
        path.write_text('{"schema_version":99}', encoding="utf-8")
        status = self.s.json("status", "--target-version", "0.2.9")
        self.assertEqual(status["classification"], "unknown_schema")


if __name__ == "__main__":
    unittest.main(verbosity=2)
