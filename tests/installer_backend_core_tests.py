#!/usr/bin/env python3
"""Hermetic contract tests for the StudioCast installer backend."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
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
                         "XDG_CACHE_HOME": str(self.root / "cache")})
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
        result, _ = self._apply_clean(False)
        self.assertEqual(result.returncode, 3)
        self.assertFalse(config.exists()); self.assertFalse((data / "models/custom/mine").exists())

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
            adapter.write_text("#!/bin/sh\n" + script, encoding="utf-8"); adapter.chmod(0o755)
            plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                               "--skip-deps", "--v4l2loopback", "--no-service", "--no-models", "--allow-unsupported")
            path = self.s.root / "helper-plan.json"; path.write_text(json.dumps(plan), encoding="utf-8")
            env = dict(base_env)
            if timeout: env["STUDIOCAST_INSTALLER_TEST_AUTH_TIMEOUT"] = timeout
            result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                                "--token", plan["approval_token"], check=False, env=env)
            return result, json.loads(result.stdout) if result.stdout.strip().startswith("{") else None

        success, value = invoke(f"cat >'{captured}'\nprintf '{{\"state\":\"succeeded\"}}\\n'\n")
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
        result, value = invoke("sleep 1\n", "0.05")
        self.assertEqual(result.returncode, 2); self.assertEqual(value["error"]["code"], "authorization_timeout")

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
