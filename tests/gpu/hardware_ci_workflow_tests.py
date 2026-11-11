#!/usr/bin/env python3
"""Hermetic checks of the actual opt-in hardware workflow commands."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
JOBS = dict(re.findall(r"^  ([a-z0-9-]+):\n(.*?)(?=^  [a-z0-9-]+:|\Z)",
                       WORKFLOW.split("jobs:\n", 1)[1], re.M | re.S))


def run_step(job, name):
    steps = re.findall(r"^      - name: (.*?)\n(.*?)(?=^      - name:|\Z)",
                       JOBS[job], re.M | re.S)
    block = dict(steps)[name]
    script = block.split("        run: |\n", 1)[1]
    return "\n".join(line[10:] for line in script.splitlines())


class HardwareWorkflowTests(unittest.TestCase):
    def test_hardware_jobs_are_explicit_dispatch_opt_ins(self):
        inputs = {"cuda-runtime-smoke": "require_cuda_runtime"}
        inputs.update({f"vulkan-runtime-smoke-{vendor}":
                       f"require_vulkan_{vendor}_runtime"
                       for vendor in ("nvidia", "amd", "intel")})
        for job, flag in inputs.items():
            with self.subTest(job=job):
                self.assertIn("if: ${{ github.event_name == 'workflow_dispatch' && "
                              f"inputs.{flag} " + "}}", JOBS[job])
                self.assertNotIn("continue-on-error", JOBS[job])
                declaration = WORKFLOW.split(f"      {flag}:\n", 1)[1]
                declaration = re.split(r"^      [a-z0-9_]+:", declaration, maxsplit=1, flags=re.M)[0]
                default = re.search(r"^        default: (.*)$", declaration, re.M)
                self.assertIsNotNone(default)
                self.assertEqual("false", default.group(1))
        for job in ("build-ubuntu", "build-mint", "build-enabled-features-no-gpu"):
            self.assertNotIn("STUDIOCAST_REQUIRE_", JOBS[job])

    def test_each_vulkan_vendor_builds_and_requires_final_resident_integration(self):
        for vendor in ("nvidia", "amd", "intel"):
            job = f"vulkan-runtime-smoke-{vendor}"
            with self.subTest(vendor=vendor):
                self.assertIn("studiocast-vulkan-final-stage-integration-tests",
                              run_step(job, "Build Vulkan tests and benchmarks"))
                script = run_step(job, "Required Vulkan final resident integration tests")
                self.assertNotIn("ctest", script)
                with tempfile.TemporaryDirectory(prefix="studiocast-ci-gates-") as directory:
                    root = Path(directory)
                    (root / "build").mkdir()
                    binary = root / "build/studiocast-vulkan-final-stage-integration-tests"
                    binary.write_text("#!/bin/bash\n"
                                      '[[ "$STUDIOCAST_REQUIRE_VULKAN_RUNTIME" == 1 ]] || exit 99\n'
                                      'exit "$FAKE_EXIT"\n', encoding="utf-8")
                    binary.chmod(0o755)
                    env = dict(os.environ, HOME=str(root), XDG_DATA_HOME=str(root / "data"))
                    for code in (0, 1, 77):
                        result = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                                cwd=root, env=dict(env, FAKE_EXIT=str(code)),
                                                capture_output=True, text=True)
                        self.assertEqual(code, result.returncode, result.stderr)
                    binary.unlink()
                    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                            cwd=root, env=env, capture_output=True, text=True)
                    self.assertNotEqual(0, result.returncode)

    def test_cuda_configuration_requires_available_dlib(self):
        script = run_step("cuda-runtime-smoke", "Configure (Open CUDA runtime validation)")
        with tempfile.TemporaryDirectory(prefix="studiocast-cuda-ci-configure-") as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            cmake = root / "bin/cmake"
            cmake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$INVOCATIONS"\n'
                             'printf "%s\\n" "$FAKE_CONFIGURE_OUTPUT"\nexit "$FAKE_EXIT"\n')
            cmake.chmod(0o755)
            calls = root / "calls"
            env = dict(os.environ, HOME=str(root), XDG_DATA_HOME=str(root / "data"),
                       PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       INVOCATIONS=str(calls))
            ready = "dlib found; Open Video Eye Contact (dlib landmarks) enabled."
            for code, output, expected in ((0, ready, 0), (1, ready, 1),
                                           (77, ready, 77), (0, "dlib not found", 1)):
                with self.subTest(code=code, output=output):
                    result = subprocess.run(["bash", "-e", "-c", script], cwd=root,
                                            env=dict(env, FAKE_EXIT=str(code),
                                                     FAKE_CONFIGURE_OUTPUT=output),
                                            capture_output=True, text=True)
                    self.assertEqual(expected, result.returncode, result.stderr)
                    self.assertIn("-DSTUDIOCAST_ENABLE_DLIB=ON", calls.read_text().splitlines())
                    self.assertIn(output, (root / "build/qualification/configure.log").read_text())

    def test_cuda_production_smoke_requires_models_and_propagates_failures(self):
        job = "cuda-runtime-smoke"
        self.assertIn("studiocast-open-cuda-production-smoke-tests",
                      run_step(job, "Build probe and production smoke"))
        script = run_step(job, "Required CUDA production smoke")
        self.assertNotIn("${{", script)  # Dispatch inputs enter through quoted environment variables.
        for suffix in ("MODEL_ROOT", "MODEL_ID", "FACE_MODEL_ID", "EYE_MODEL_ID",
                       "DENOISE_MODEL_ID", "REPLACE_PATH"):
            self.assertIn("CUDA_" + suffix + ": ${{ inputs.cuda_" + suffix.lower() + " }}",
                          JOBS[job])
            self.assertIn("      cuda_" + suffix.lower() + ":\n", WORKFLOW)
        self.assertIn("if: ${{ always() }}", JOBS[job])
        self.assertIn("path: build/qualification/", JOBS[job])
        effects = ["blur", "keylight", "blur+keylight", "vignette",
                   "blur+vignette", "keylight+vignette", "blur+keylight+vignette",
                   "denoise", "auto_frame", "eye_contact", "mirror", "remove", "replace",
                   "auto_frame+tracked_vignette",
                   "denoise+auto_frame+eye_contact+blur+keylight+tracked_vignette+mirror"]
        cases = [("cuda", effect) for effect in effects]
        cases.append(("auto", "denoise+auto_frame+eye_contact+blur+keylight+tracked_vignette"))
        with tempfile.TemporaryDirectory(prefix="studiocast-cuda-ci-gates-") as directory:
            root = Path(directory)
            (root / "build").mkdir()
            (root / "bin").mkdir()
            (root / "build/CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\nSTUDIOCAST_ENABLE_OPEN_CUDA:BOOL=ON\n"
                "STUDIOCAST_ENABLE_DLIB:BOOL=ON\n")
            for command in ("git", "nvidia-smi"):
                fake = root / "bin" / command
                fake.write_text("#!/bin/bash\nprintf 'fixture evidence\\n'\n")
                fake.chmod(0o755)
            binary = root / "build/studiocast-open-cuda-production-smoke-tests"
            binary.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$INVOCATIONS"\n'
                              'selection="$2"\nwhile (($#)); do\n'
                              '  if [[ "$1" == --effects ]]; then effects="$2"; fi\n'
                              '  shift\ndone\n'
                              'if [[ "$FAKE_FAIL_SELECTION" == "$selection" && '
                              '"$FAKE_FAIL_EFFECT" == "$effects" ]]; then exit "$FAKE_EXIT"; fi\n')
            binary.chmod(0o755)
            model_root = root / "model root $(touch escaped)/studiocast/models/open_video"
            model_root.mkdir(parents=True)
            replacement = root / "replacement $(touch escaped); 'image'.png"
            replacement.write_bytes(b"preinstalled image fixture")
            calls = root / "calls"
            env = dict(os.environ, HOME=str(root), XDG_DATA_HOME=str(root / "data"),
                       PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       CUDA_MODEL_ROOT=str(model_root),
                       CUDA_MODEL_ID="model $(touch escaped)",
                       CUDA_FACE_MODEL_ID="face `touch escaped`",
                       CUDA_EYE_MODEL_ID="eye'; touch escaped; #\nsecond line",
                       CUDA_DENOISE_MODEL_ID='denoise "$(touch escaped)"',
                       CUDA_REPLACE_PATH=str(replacement), INVOCATIONS=str(calls),
                       FAKE_FAIL_SELECTION="", FAKE_FAIL_EFFECT="", FAKE_EXIT="0")

            def invoke(**overrides):
                calls.unlink(missing_ok=True)
                return subprocess.run(["bash", "-e", "-c", script], cwd=root,
                                      env=dict(env, **overrides), capture_output=True, text=True)

            # Failures after both original cases and the automatic branch must
            # survive tee and stop subsequent cases, including skip status 77.
            outcomes = [(0, "", "", len(cases))]
            for code in (1, 77):
                outcomes.extend((code, "--" + selection, effect, index + 1)
                                for index, (selection, effect) in enumerate(cases)
                                if effect in ("blur", "denoise", cases[-1][1]))
            for code, selection, effect, count in outcomes:
                with self.subTest(code=code, selection=selection, effect=effect):
                    result = invoke(FAKE_EXIT=str(code), FAKE_FAIL_SELECTION=selection,
                                    FAKE_FAIL_EFFECT=effect)
                    self.assertEqual(code, result.returncode, result.stderr)
                    arguments = calls.read_bytes().decode().split("\0")[:-1]
                    expected = []
                    for case_selection, case_effect in cases[:count]:
                        expected.extend(["--require", "--" + case_selection,
                                         "--model-root", str(model_root),
                                         "--model-id", env["CUDA_MODEL_ID"],
                                         "--face-model-id", env["CUDA_FACE_MODEL_ID"],
                                         "--eye-model-id", env["CUDA_EYE_MODEL_ID"],
                                         "--denoise-model-id", env["CUDA_DENOISE_MODEL_ID"],
                                         "--replace-path", str(replacement), "--effects", case_effect,
                                         "--input", "64x48", "--output", "32x24", "--frames", "3"])
                        self.assertTrue((root / "build/qualification" /
                                         f"{case_selection}-{case_effect}.log").is_file())
                    self.assertEqual(expected, arguments)
                    self.assertFalse((root / "escaped").exists())
            missing = [{"CUDA_" + suffix: ""} for suffix in
                       ("MODEL_ROOT", "MODEL_ID", "FACE_MODEL_ID", "EYE_MODEL_ID",
                        "DENOISE_MODEL_ID", "REPLACE_PATH")]
            missing.extend([{"CUDA_MODEL_ROOT": str(root / "absent")},
                            {"CUDA_REPLACE_PATH": str(root / "absent.png")},
                            {"CUDA_REPLACE_PATH": str(root)}])
            for overrides in missing:
                with self.subTest(missing=overrides):
                    result = invoke(**overrides)
                    self.assertNotEqual(0, result.returncode)
                    self.assertFalse(calls.exists())
            binary.unlink()
            self.assertNotEqual(0, invoke().returncode)
            self.assertFalse(calls.exists())


if __name__ == "__main__":
    unittest.main()
