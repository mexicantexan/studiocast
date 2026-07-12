#!/usr/bin/env python3
"""Hermetic AppDir/release integration tests; no network or host mutation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "installer/backend/studiocast-installer-backend"
FIXTURES = ROOT / "tests/data/installer_release"
KEY_ID = "fixture-ed25519-2026"
DEFAULT_PACKS = [
    "fastenhancer_s_vd_v1", "fastenhancer_m_vd_v1", "modnet-webnn-256-fp32",
    "yunet_opencv_zoo_2023mar_fp32", "dlib_68_ibug_300w",
    "gaze_correction_cam_flx_v0_1_1", "fastdvdnet_sigma15",
]


def facts() -> dict:
    return {
        "schema_version": 1, "facts_version": "installer-facts/v1",
        "os": {"id": "ubuntu", "version_id": "24.04", "base_id": "ubuntu",
               "base_version_id": "24.04", "architecture": "x86_64", "kernel_release": "fixture",
               "supported": True, "reason_codes": ["os.supported.ubuntu_noble"]},
        "installation": {"classification": "absent", "active_version": None, "previous_version": None,
                         "target_version": "0.3.0", "version_relation": "not_installed",
                         "desired_configuration": None, "reason_codes": ["install.manifest.absent"]},
        "paths": {"free_bytes": 20_000_000_000,
                  "required_bytes": {"download": 1, "build": 1, "staging": 1}},
        "toolchain": {"tools": {}, "missing_build_dependencies": [], "reason_codes": []},
        "runtime": {"onnxruntime": {"present": False, "compatible": False,
                    "providers": {"cpu": False, "cuda": False}, "probe": "not_run"}},
        "systemd_user": {"systemctl_present": False, "manager_usable": False, "reason_codes": []},
        "v4l2": {"module_available": True, "module_loaded": True, "device_present": True,
                  "device_number_conflict": False, "persistence_state": "configured",
                  "kernel_headers_usable": True, "dkms_viable": True},
        "privileged_helper": {"present": False, "trusted": False, "compatible": False},
        "gpus": {"devices": [], "nvidia": {"driver_usable": False, "cuda_usable": False},
                 "vulkan": {"loader_present": False, "physical_devices": [], "compute_device_usable": False}},
        "maxine": {"components": {}, "effects": {}, "reason_codes": []},
        "effects": {"capabilities": {}}, "models": {"packs": {}, "default_pack_ids": []},
        "cache": {"release_artifacts": {}, "model_artifacts": {}},
        "connectivity": {"release_source": {"state": "online", "reason_codes": []},
                         "model_source": {"state": "online", "reason_codes": []}}, "reason_codes": [],
    }


class PackagingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="studiocast-packaging-")
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.env = os.environ.copy()
        self.env.update({"HOME": str(self.home), "XDG_DATA_HOME": str(self.root / "data"),
                         "XDG_CONFIG_HOME": str(self.root / "config"), "XDG_STATE_HOME": str(self.root / "state"),
                         "XDG_CACHE_HOME": str(self.root / "cache"), "STUDIOCAST_INSTALLER_TEST_MODE": "1"})
        self.fixture = self.root / "fixture"
        shutil.copytree(FIXTURES, self.fixture)
        installer = self.root / "AppDir/usr/share/studiocast/installer"
        installer.mkdir(parents=True)
        self.backend_path = installer / "studiocast-installer-backend"
        shutil.copy2(BACKEND, self.backend_path)
        self.backend_path.chmod(0o755)
        shutil.copytree(ROOT / "packaging/release", installer / "release",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(ROOT / "installer/models", installer / "models",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        component_root = installer.parent
        model_catalog = component_root / "packaging/models"
        model_catalog.mkdir(parents=True)
        shutil.copy2(ROOT / "packaging/models/curated-model-catalog-v1.json", model_catalog)
        shutil.copytree(ROOT / "resources/model_packs", component_root / "resources/model_packs",
                        ignore=shutil.ignore_patterns("*.onnx", "*.dat", "*.bin"))
        trust = installer / "trust/keys"
        trust.mkdir(parents=True)
        shutil.copy2(self.fixture / "fixture-ed25519-public.pem", trust / f"{KEY_ID}.pem")
        # Exercise a fixture installer that meets the signed manifest's stated
        # minimum so source acquisition can proceed after Apply.
        (self.root / "AppDir/usr/share/VERSION").write_text("0.2.9\n")
        self.facts_path = self.root / "facts.json"
        self.facts_path.write_text(json.dumps(facts()), encoding="utf-8")

    def tearDown(self) -> None: self.temporary.cleanup()

    def backend(self, *args: str, check: bool = True, production: bool = False) -> subprocess.CompletedProcess[str]:
        env = dict(self.env)
        if production: env.pop("STUDIOCAST_INSTALLER_TEST_MODE", None)
        result = subprocess.run([str(BACKEND if production else self.backend_path), *args], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and result.returncode:
            self.fail(f"backend failed ({result.returncode})\nstdout={result.stdout}\nstderr={result.stderr}")
        return result

    def verified_receipt(self) -> tuple[dict, Path]:
        receipt = self.root / "receipt.json"
        value = json.loads(self.backend(
            "verify-release", "--release-manifest", str(self.fixture / "manifest.json"),
            "--release-signature", str(self.fixture / "manifest.json.sig"),
            "--release-archive", str(self.fixture / "source.tar.gz"),
            "--release-receipt-out", str(receipt),
            "--release-cache-dir", str(self.root / "cache/studiocast/release")).stdout)
        return value, receipt

    @staticmethod
    def terminal_result(text: str) -> dict:
        marker = text.rfind("\n{")
        return json.loads(text[marker + 1 if marker >= 0 else 0:])

    def test_installer_component_stages_release_primitives_and_production_trust_contract(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        for required in ("packaging/release/release_channel.py", "release-manifest-v1.schema.json",
                         "packaging/release/keys/README.md", "installer/models/studiocast-model-transaction",
                         "installer/models/model_transactions.py", "curated-model-catalog-v1.json",
                         "resources/model_packs/", "installer/privileged/client_contract.py"):
            self.assertIn(required, cmake)
        verifier = (ROOT / "packaging/appimage/verify_bundle.sh").read_text(encoding="utf-8")
        self.assertIn("installer/privileged/client_contract.py", verifier)
        self.assertIn('PATTERN "model.json"', cmake)
        self.assertIn('PATTERN "LICENSE.txt"', cmake)
        for forbidden_pattern in ('PATTERN "*.onnx"', 'PATTERN "*.dat"', 'PATTERN "*.bin"'):
            self.assertNotIn(forbidden_pattern, cmake)
        self.assertEqual([], list((ROOT / "packaging/release/keys").glob("*.pem")))
        self.assertNotIn("fixture-ed25519-public.pem", cmake)
        packaging_script = (ROOT / "packaging/appimage/build_appimage.sh").read_text(encoding="utf-8")
        self.assertIn("--trusted-release-key", packaging_script)
        self.assertNotIn("falling back to a working-tree tarball", packaging_script)
        self.assertNotIn("find \"${REPO_ROOT}\" \"${DIST_DIR}\"", packaging_script)
        self.assertIn("official bundles require a clean worktree", packaging_script)

        missing_key = subprocess.run(
            [str(ROOT / "packaging/appimage/build_appimage.sh"), "--dry-run", "--appimage-required"],
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(2, missing_key.returncode)
        self.assertIn("require at least one --trusted-release-key", missing_key.stderr)

        contradictory = subprocess.run(
            [
                str(ROOT / "packaging/appimage/build_appimage.sh"), "--dry-run",
                "--appimage-required", "--skip-appimage",
            ],
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(2, contradictory.returncode)
        self.assertIn("cannot be combined", contradictory.stderr)

        fixture_key = subprocess.run(
            [
                str(ROOT / "packaging/appimage/build_appimage.sh"), "--dry-run", "--skip-appimage",
                "--trusted-release-key", f"fixture-key={self.fixture / 'fixture-ed25519-public.pem'}",
            ],
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(2, fixture_key.returncode)
        self.assertIn("test/fixture", fixture_key.stderr)

    def test_valid_receipt_is_bound_and_reverified_for_recommended_plan(self) -> None:
        receipt, path = self.verified_receipt()
        self.assertEqual("studiocast-verified-release/v1", receipt["receipt_version"])
        self.assertEqual((self.fixture / "source.tar.gz").stat().st_size, receipt["archive_size"])
        plan = json.loads(self.backend(
            "plan", "install", "--json", "--facts", str(self.facts_path),
            "--release-receipt", str(path),
            "--no-v4l2loopback", "--no-service", "--no-models").stdout)
        codes = {item["code"] for item in plan["blockers"]}
        self.assertNotIn("source.signed_verification_receipt_required", codes)
        self.assertNotIn("release.installer_update_required", codes)
        self.assertTrue(plan["source"]["official"])
        self.assertEqual(receipt["archive_sha256"], plan["source"]["archive_sha256"])
        self.assertEqual("studiocast-source", plan["downloads"][0]["artifact_id"])
        self.assertTrue(plan["downloads"][0]["required_for_core"])

    def test_gui_shaped_verified_plan_stays_recommended_with_explicit_selections(self) -> None:
        receipt, path = self.verified_receipt()
        build_dir = self.root / "cache/studiocast/builds/0.3.0"
        model_destination = self.root / "data/studiocast/models"
        args = ["plan", "install", "--facts", str(self.facts_path),
            "--release-receipt", str(path), "--build-dir", str(build_dir),
            "--build-type", "Release", "--skip-deps", "--v4l2loopback",
            "--v4l-device-number", "10", "--v4l-label", "StudioCast Camera",
            "--v4l-exclusive-caps", "--no-service", "--open-cuda", "--open-audio",
            "--no-open-vulkan", "--models", "--model-destination", str(model_destination)]
        for pack_id in DEFAULT_PACKS:
            args += ["--model-id", pack_id]
        plan = json.loads(self.backend(*args).stdout)
        self.assertEqual(plan["route"], "recommended")
        self.assertTrue(plan["source"]["official"])
        self.assertEqual(plan["desired_state"]["model_pack_ids"], DEFAULT_PACKS)
        self.assertEqual(plan["desired_state"]["model_destination"], str(model_destination))
        self.assertEqual(plan["paths"]["build_cache"], str(build_dir))
        self.assertTrue(plan["desired_state"]["virtual_camera"]["required_for_success"])
        self.assertEqual(plan["desired_state"]["features"], {
            "open_audio": True, "open_cuda": True, "open_vulkan": False})
        self.assertNotIn("source.signed_verification_receipt_required",
                         {item["code"] for item in plan["blockers"]})
        self.assertEqual(len([item for item in plan["downloads"]
                              if not item["required_for_core"]]), 8)

    def test_arbitrary_archive_routes_advanced_without_becoming_official(self) -> None:
        plan = json.loads(self.backend("plan", "install", "--json", "--facts", str(self.facts_path),
                                   "--release-archive", str(self.fixture / "source.tar.gz"),
                                   "--official-source", "--no-v4l2loopback", "--no-service", "--no-models").stdout)
        self.assertFalse(plan["source"]["official"])
        self.assertEqual("advanced", plan["route"])
        self.assertNotIn("source.signed_verification_receipt_required", {x["code"] for x in plan["blockers"]})
        failed = self.backend("verify-release", "--release-manifest", str(self.fixture / "manifest.json"),
                          "--release-signature", str(self.fixture / "manifest.json.sig"),
                          "--release-archive", str(self.fixture / "source.tar.gz"), check=False, production=True)
        self.assertEqual(2, failed.returncode)
        self.assertIn("release.signature.production_key_missing", failed.stderr)

    def test_manifest_signature_source_hash_and_receipt_tamper_fail(self) -> None:
        bad_signature = self.fixture / "bad.sig"; bad_signature.write_bytes(b"{}")
        failed = self.backend("verify-release", "--release-manifest", str(self.fixture / "manifest.json"),
                          "--release-signature", str(bad_signature), "--release-archive", str(self.fixture / "source.tar.gz"),
                          check=False)
        self.assertEqual(2, failed.returncode)
        receipt, path = self.verified_receipt()
        (self.fixture / "source.tar.gz").write_bytes(b"corrupt")
        failed = self.backend("verify-release", "--release-receipt", str(path), check=False)
        self.assertEqual(2, failed.returncode)
        self.assertRegex(failed.stderr, "release.artifact.(size|hash)_mismatch")
        receipt["archive_path"] = "/tmp/not-the-signed-artifact"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        failed = self.backend("verify-release", "--release-receipt", str(path), check=False)
        self.assertEqual(2, failed.returncode)

    def test_configurable_local_channel_offline_receipt_and_self_update_offer(self) -> None:
        manifest_url = (self.fixture / "manifest.json").resolve().as_uri()
        signature_url = (self.fixture / "manifest.json.sig").resolve().as_uri()
        status = json.loads(self.backend(
            "release-status", "--stable-url", manifest_url, "--stable-signature-url", signature_url,
            "--release-cache-dir", str(self.root / "stable-cache"),
            "--release-archive", str(self.fixture / "source.tar.gz"),
            "--installed-version", "0.2.9").stdout)
        self.assertEqual("update", status["action"])
        receipt, path = self.verified_receipt()
        base = ["release-status", "--release-receipt", str(path)]
        same = json.loads(self.backend(*base, "--installed-version", "0.3.0").stdout)
        self.assertEqual(("keep_current", "same"), (same["action"], same["relation"]))
        reinstall = json.loads(self.backend(*base, "--installed-version", "0.3.0", "--allow-same-version").stdout)
        self.assertEqual("reinstall", reinstall["action"])
        newer = json.loads(self.backend(*base, "--installed-version", "0.4.0").stdout)
        self.assertEqual(("keep_current", "installed_newer"), (newer["action"], newer["relation"]))
        downgrade = json.loads(self.backend(*base, "--installed-version", "0.4.0", "--allow-downgrade").stdout)
        self.assertEqual("downgrade", downgrade["action"])
        running = self.root / "running.AppImage"; running.write_bytes(b"running stays unchanged")
        before = running.read_bytes()
        installer_cache = self.root / "offline-cache/installer"
        installer_cache.mkdir(parents=True)
        shutil.copy2(self.fixture / "installer.AppImage", installer_cache / "installer.AppImage")
        offline = json.loads(self.backend(
            "release-status", "--release-receipt", str(path),
            "--release-cache-dir", str(self.root / "offline-cache"), "--installed-version", "0.2.9",
            "--prepare-self-update", "--appimage-path", str(running)).stdout)
        self.assertEqual("offer_restart", offline["self_update"]["state"])
        self.assertTrue(offline["self_update"]["requires_confirmation"])
        self.assertFalse(offline["self_update"]["running_artifact_replaced"])
        self.assertEqual(before, running.read_bytes())
        self.assertNotEqual(str(running), offline["self_update"]["verified_appimage"])

    def test_release_metadata_is_read_only_and_source_acquisition_is_reviewed_apply_work(self) -> None:
        metadata_cache = self.root / "metadata-only-cache"
        status = json.loads(self.backend(
            "release-status", "--stable-url", (self.fixture / "manifest.json").resolve().as_uri(),
            "--stable-signature-url", (self.fixture / "manifest.json.sig").resolve().as_uri(),
            "--release-cache-dir", str(metadata_cache), "--installed-version", "0.2.9").stdout)
        self.assertEqual(status["source_archive_state"], "acquisition_pending")
        self.assertTrue(status["source_acquisition_required"])
        self.assertFalse((metadata_cache / "artifacts/source.tar.gz").exists())

        receipt, receipt_path = self.verified_receipt()
        cache_path = Path(receipt["archive_path"])
        self.assertFalse(cache_path.exists())
        plan = json.loads(self.backend("plan", "install", "--facts", str(self.facts_path),
            "--release-receipt", str(receipt_path), "--no-v4l2loopback", "--no-service",
            "--no-models", "--release-cache-dir", str(cache_path.parent.parent)).stdout)
        acquire = next(op for op in plan["operations"] if op["kind"] == "release_acquire")
        self.assertEqual(acquire["id"], "source.release.acquire")
        self.assertEqual(acquire["inputs"]["artifact"]["sha256"], receipt["archive_sha256"])
        plan_path = self.root / "release-plan.json"; plan_path.write_text(json.dumps(plan))
        applied = self.backend("apply-plan", "--plan", str(plan_path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.facts_path), check=False)
        terminal = self.terminal_result(applied.stdout)
        self.assertEqual(terminal["error"]["code"], "archive.unsupported")
        self.assertEqual(cache_path.read_bytes(), (self.fixture / "source.tar.gz").read_bytes())
        journal = json.loads((self.root / "data/studiocast/install-manifest.json").read_text())["journal"]
        self.assertEqual([(item["operation_id"], item["state"]) for item in journal[:3]],
                         [("preflight.validate", "completed"),
                          ("source.release.acquire", "completed"),
                          ("source.archive.extract", "failed")])

    def test_release_acquire_offline_hit_miss_and_bad_candidate_preserve_good_cache(self) -> None:
        receipt, receipt_path = self.verified_receipt()
        cache_path = Path(receipt["archive_path"])
        cache_path.parent.mkdir(parents=True)
        shutil.copy2(self.fixture / "source.tar.gz", cache_path)
        good = cache_path.read_bytes()

        def reset_install_state() -> None:
            data = self.root / "data/studiocast"
            if data.exists(): shutil.rmtree(data)

        # The candidate may change after review; an already verified cache is
        # retained and used without replacing it from the bad candidate.
        plan = json.loads(self.backend("plan", "install", "--facts", str(self.facts_path),
            "--release-receipt", str(receipt_path), "--no-v4l2loopback", "--no-service",
            "--no-models", "--release-cache-dir", str(cache_path.parent.parent)).stdout)
        (self.fixture / "source.tar.gz").write_bytes(b"changed after review")
        path = self.root / "bad-candidate-plan.json"; path.write_text(json.dumps(plan))
        result = self.backend("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.facts_path), check=False)
        self.assertEqual(self.terminal_result(result.stdout)["error"]["code"], "archive.unsupported")
        self.assertEqual(cache_path.read_bytes(), good)

        # A metadata-only receipt can use the verified content-addressed cache offline.
        reset_install_state()
        pending = self.root / "pending-receipt.json"
        metadata = json.loads(self.backend("verify-release", "--release-manifest", str(self.fixture / "manifest.json"),
            "--release-signature", str(self.fixture / "manifest.json.sig"),
            "--release-receipt-out", str(pending), "--release-cache-dir", str(cache_path.parent.parent)).stdout)
        self.assertEqual(metadata["archive_state"], "acquisition_pending")
        plan = json.loads(self.backend("plan", "install", "--facts", str(self.facts_path),
            "--release-receipt", str(pending), "--offline", "--no-v4l2loopback", "--no-service",
            "--no-models", "--release-cache-dir", str(cache_path.parent.parent)).stdout)
        path.write_text(json.dumps(plan))
        hit = self.backend("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.facts_path), check=False)
        self.assertEqual(self.terminal_result(hit.stdout)["error"]["code"], "archive.unsupported")

        reset_install_state()
        cache_path.unlink()
        plan = json.loads(self.backend("plan", "install", "--facts", str(self.facts_path),
            "--release-receipt", str(pending), "--offline", "--no-v4l2loopback", "--no-service",
            "--no-models", "--release-cache-dir", str(cache_path.parent.parent)).stdout)
        path.write_text(json.dumps(plan))
        miss = self.backend("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.facts_path), check=False)
        self.assertEqual(self.terminal_result(miss.stdout)["error"]["code"], "release.cache.miss")
        self.assertFalse(cache_path.exists())

    def test_terminal_result_is_bound_to_reviewed_plan(self) -> None:
        plan = json.loads(self.backend("plan", "install", "--json", "--facts", str(self.facts_path),
                                   "--source-dir", str(ROOT), "--no-v4l2loopback", "--no-service",
                                   "--no-models", "--allow-unsupported").stdout)
        plan_path = self.root / "plan.json"; plan_path.write_text(json.dumps(plan), encoding="utf-8")
        # Dry-run emits reviewed commands before its terminal JSON; parse the
        # final JSON object by locating the result-version marker.
        applied = self.backend("apply-plan", "--plan", str(plan_path), "--digest", plan["plan_digest"],
                           "--token", plan["approval_token"], "--facts", str(self.facts_path), "--dry-run")
        marker = applied.stdout.rfind("\n{") + 1
        terminal = json.loads(applied.stdout[marker:])
        self.assertEqual(1, terminal["schema_version"])
        self.assertEqual("installer-result/v1", terminal["result_version"])
        self.assertEqual(plan["plan_id"], terminal["plan_id"])
        self.assertEqual(plan["plan_digest"], terminal["plan_digest"])
        self.assertEqual(terminal["state"], terminal["transaction_state"])
        plan["operations"][0]["id"] = "tampered"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        rejected = self.backend("apply-plan", "--plan", str(plan_path), "--digest", plan["plan_digest"],
                                "--token", plan["approval_token"], check=False)
        failure = json.loads(rejected.stdout)
        self.assertEqual("failed", failure["transaction_state"])
        self.assertEqual(plan["plan_id"], failure["plan_id"])
        self.assertEqual(plan["plan_digest"], failure["plan_digest"])
        self.assertEqual("plan.digest_mismatch", failure["error"]["code"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
