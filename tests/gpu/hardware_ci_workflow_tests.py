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
        inputs["vulkan-matting-production"] = "require_vulkan_matting_runtime"
        inputs["vulkan-model-effects-production"] = "require_vulkan_model_effects_runtime"
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

    def test_vulkan_matting_requires_models_and_propagates_native_failures(self):
        job = "vulkan-matting-production"
        self.assertIn("-DSTUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=ON",
                      run_step(job, "Configure production Vulkan matting"))
        self.assertIn('"${VULKAN_NCNN_PREFIX}"',
                      run_step(job, "Configure production Vulkan matting"))
        self.assertIn("studiocast-vulkan-matting-production-smoke",
                      run_step(job, "Build production Vulkan matting smoke"))
        script = run_step(job, "Required production Vulkan matting smoke")
        self.assertNotIn("${{", script)
        self.assertIn("if: ${{ always() }}", JOBS[job])
        self.assertNotIn("curl ", JOBS[job])
        self.assertNotIn("apt-get", JOBS[job])
        with tempfile.TemporaryDirectory(prefix="studiocast-vulkan-matting-ci-") as directory:
            root = Path(directory)
            (root / "build").mkdir()
            model_root = root / "model $(touch escaped)"
            model_root.mkdir()
            replacement = root / "replacement '$(touch escaped)'.png"
            replacement.write_bytes(b"fixture")
            binary = root / "build/studiocast-vulkan-matting-production-smoke"
            binary.write_text('#!/bin/bash\n'
                              '[[ "$STUDIOCAST_REQUIRE_VULKAN_RUNTIME" == 1 ]] || exit 99\n'
                              'printf "%s\\0" "$@" > "$INVOCATIONS"\nexit "$FAKE_EXIT"\n')
            binary.chmod(0o755)
            calls = root / "calls"
            env = dict(os.environ, VULKAN_MODEL_ROOT=str(model_root), VULKAN_MODEL_ID="model; false",
                       VULKAN_REPLACE_PATH=str(replacement), RUNNER_TEMP=str(root), INVOCATIONS=str(calls))
            for code in (0, 1, 77):
                with self.subTest(code=code):
                    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                            cwd=root, env=env | {"FAKE_EXIT": str(code)},
                                            capture_output=True, text=True)
                    self.assertEqual(code, result.returncode, result.stderr)
                    self.assertEqual(["--require", "--model-root", str(model_root), "--model-id",
                                      env["VULKAN_MODEL_ID"], "--replace-path", str(replacement),
                                      "--frames", "3"], calls.read_bytes().decode().split("\0")[:-1])
                    self.assertFalse((root / "escaped").exists())
            for field in ("VULKAN_MODEL_ROOT", "VULKAN_MODEL_ID", "VULKAN_REPLACE_PATH"):
                calls.unlink(missing_ok=True)
                result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=root,
                                        env=env | {field: "", "FAKE_EXIT": "0"},
                                        capture_output=True, text=True)
                self.assertNotEqual(0, result.returncode)
                self.assertFalse(calls.exists())
            binary.unlink()
            self.assertNotEqual(0, subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script], cwd=root, env=env,
                capture_output=True, text=True).returncode)

    def vulkan_models_fixture(self, root):
        (root / "bin").mkdir()
        (root / "build/qualification").mkdir(parents=True)
        sdk = root / "sdk $(touch escaped); 'prefix'"
        sdk_files = ("lib/cmake/ncnn/ncnnConfig.cmake", "lib/libncnn.a",
                     "share/licenses/studiocast-ncnn/dependency.json",
                     "share/licenses/studiocast-ncnn/ncnn-LICENSE.txt",
                     "share/licenses/studiocast-ncnn/glslang-LICENSE.txt")
        for relative in sdk_files:
            path = sdk / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"preinstalled SDK fixture")
        model_root = root / "model root $(touch escaped)/studiocast/models/open_video"
        model_root.mkdir(parents=True)
        face = root / "face $(touch escaped); 'local'.jpg"
        replacement = root / "replacement '$(touch escaped)'.png"
        face.write_bytes(b"local face fixture")
        replacement.write_bytes(b"preinstalled replacement fixture")
        runner_temp = root / "runner temp"
        runner_temp.mkdir()
        env = dict(os.environ, HOME=str(root / "original-home"),
                   XDG_DATA_HOME=str(root / "original-data"),
                   XDG_CONFIG_HOME=str(root / "original-config"),
                   XDG_CACHE_HOME=str(root / "original-cache"),
                   XDG_STATE_HOME=str(root / "original-state"),
                   PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                   RUNNER_TEMP=str(runner_temp), VULKAN_NCNN_PREFIX=str(sdk),
                   VULKAN_MODEL_ROOT=str(model_root), VULKAN_MODEL_ID="model'; touch escaped; #\nsecond line",
                   VULKAN_REPLACE_PATH=str(replacement), VULKAN_FACE_PATH=str(face),
                   CMAKE_INVOCATIONS=str(root / "cmake-calls"), SMOKE_INVOCATIONS=str(root / "smoke-calls"),
                   DIAGNOSTICS_ENV=str(root / "diagnostics-env"), SMOKE_ENV=str(root / "smoke-env"),
                   FAKE_CONFIGURE_EXIT="0", FAKE_BUILD_EXIT="0", FAKE_DLIB_RECEIPT="TRUE",
                   FAKE_DIAGNOSTICS_EXIT="0", FAKE_SMOKE_EXIT="0", FAKE_INVENTORY_EXIT="0")
        scripts = {
            "bin/cmake": '#!/bin/bash\nprintf "%s\\0" "$@" > "$CMAKE_INVOCATIONS"\n'
                'if [[ "${1:-}" == --build ]]; then exit "$FAKE_BUILD_EXIT"; fi\n'
                'printf "CMAKE_BUILD_TYPE:STRING=Release\\nSTUDIOCAST_DLIB_AVAILABLE:INTERNAL=%s\\n" '
                '"$FAKE_DLIB_RECEIPT" > build/CMakeCache.txt\n'
                'printf "fixture configure output\\n"\nexit "$FAKE_CONFIGURE_EXIT"\n',
            "bin/git": '#!/bin/bash\nprintf "fixture source SHA\\n"\n',
            "bin/vulkaninfo": '#!/bin/bash\nprintf "fixture Vulkan device evidence\\n"\nexit "$FAKE_INVENTORY_EXIT"\n',
            "build/studiocast-vulkan-model-diagnostics-tests": '#!/bin/bash\n'
                '[[ "$STUDIOCAST_REQUIRE_VULKAN_MODEL_DIAGNOSTICS" == 1 ]] || exit 99\n'
                'printf "%s\\0" "$HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" '
                '"$XDG_STATE_HOME" > "$DIAGNOSTICS_ENV"\n'
                'printf "fixture native cold diagnostics\\n"\nexit "$FAKE_DIAGNOSTICS_EXIT"\n',
            "build/studiocast-vulkan-effects-production-smoke": '#!/bin/bash\n'
                '[[ "$STUDIOCAST_REQUIRE_VULKAN_RUNTIME" == 1 ]] || exit 99\n'
                'printf "%s\\0" "$@" > "$SMOKE_INVOCATIONS"\n'
                'printf "%s\\0" "$HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" '
                '"$XDG_STATE_HOME" > "$SMOKE_ENV"\n'
                'printf "fixture native full matrix\\n"\nexit "$FAKE_SMOKE_EXIT"\n',
        }
        for relative, script in scripts.items():
            path = root / relative
            path.write_text(script)
            path.chmod(0o755)
        return env, sdk_files

    def test_full_vulkan_models_job_uses_only_preinstalled_inputs_and_exact_dlib_receipt(self):
        job = "vulkan-model-effects-production"
        block = JOBS[job]
        for forbidden in ("apt-get", "curl ", "wget ", "scripts/setup.sh", "continue-on-error"):
            self.assertNotIn(forbidden, block)
        self.assertIn("inputs.vulkan_matting_runner_labels", block)
        for variable, input_name in (("VULKAN_NCNN_PREFIX", "vulkan_ncnn_prefix"),
                                     ("VULKAN_MODEL_ROOT", "vulkan_matting_model_root"),
                                     ("VULKAN_MODEL_ID", "vulkan_matting_model_id"),
                                     ("VULKAN_REPLACE_PATH", "vulkan_matting_replace_path"),
                                     ("VULKAN_FACE_PATH", "vulkan_model_effects_face_path")):
            self.assertIn(variable + ": ${{ inputs." + input_name + " }}", block)
        self.assertIn("if: ${{ always() }}", block)
        self.assertIn("path: build/qualification/", block)
        configure = run_step(job, "Configure production Vulkan model effects")
        self.assertNotIn("${{", configure)
        with tempfile.TemporaryDirectory(prefix="studiocast-vulkan-models-configure-") as directory:
            root = Path(directory)
            env, sdk_files = self.vulkan_models_fixture(root)
            calls = root / "cmake-calls"
            def invoke(**overrides):
                calls.unlink(missing_ok=True)
                return subprocess.run(["bash", "-c", configure], cwd=root, env=env | overrides,
                                      capture_output=True, text=True)
            for code in (0, 1, 77):
                with self.subTest(configure_exit=code):
                    result = invoke(FAKE_CONFIGURE_EXIT=str(code))
                    self.assertEqual(code, result.returncode, result.stderr)
                    arguments = calls.read_bytes().decode().split("\0")[:-1]
                    self.assertIn("-Dncnn_DIR=" + env["VULKAN_NCNN_PREFIX"] + "/lib/cmake/ncnn", arguments)
                    for flag in ("OPEN_CUDA=OFF", "CUDA_KERNELS=OFF", "OPEN_AUDIO=OFF", "DLIB=ON",
                                 "OPEN_VULKAN=ON", "NCNN_VULKAN_MATTING=ON"):
                        self.assertIn("-DSTUDIOCAST_ENABLE_" + flag, arguments)
                    self.assertIn("-DCMAKE_BUILD_TYPE=Release", arguments)
                    self.assertIn("fixture configure output", (root / "build/qualification/configure.log").read_text())
                    self.assertFalse((root / "escaped").exists())
            for receipt in ("", "FALSE", "OFF", "0", "true", "NOTFOUND"):
                with self.subTest(missing_dlib=receipt):
                    self.assertNotEqual(0, invoke(FAKE_DLIB_RECEIPT=receipt).returncode)
            for receipt in ("TRUE", "ON", "1"):
                self.assertEqual(0, invoke(FAKE_DLIB_RECEIPT=receipt).returncode)
            missing = [{key: ""} for key in ("VULKAN_NCNN_PREFIX", "VULKAN_MODEL_ROOT", "VULKAN_MODEL_ID",
                                             "VULKAN_REPLACE_PATH", "VULKAN_FACE_PATH")]
            wrong_root = root / "wrong-shape"
            wrong_root.mkdir()
            missing += [{"VULKAN_NCNN_PREFIX": str(root / "absent-sdk")},
                        {"VULKAN_MODEL_ROOT": str(root / "absent-models")},
                        {"VULKAN_MODEL_ROOT": str(wrong_root)},
                        {"VULKAN_FACE_PATH": str(root / "absent.jpg")},
                        {"VULKAN_FACE_PATH": str(root)},
                        {"VULKAN_REPLACE_PATH": str(root / "absent.png")},
                        {"VULKAN_REPLACE_PATH": str(root)}]
            for overrides in missing:
                with self.subTest(missing=overrides):
                    self.assertNotEqual(0, invoke(**overrides).returncode)
                    self.assertFalse(calls.exists())
            for relative in sdk_files:
                source = Path(env["VULKAN_NCNN_PREFIX"]) / relative
                saved = source.read_bytes()
                source.unlink()
                with self.subTest(missing_sdk_file=relative):
                    self.assertNotEqual(0, invoke().returncode)
                    self.assertFalse(calls.exists())
                source.write_bytes(saved)
            self.assertTrue((root / "build/qualification/source.txt").is_file())
            self.assertTrue((root / "build/qualification/preinstalled-inputs.sha256").is_file())

    def test_full_vulkan_models_build_and_native_steps_propagate_every_failure(self):
        job = "vulkan-model-effects-production"
        build = run_step(job, "Build production Vulkan model effects validation")
        native = run_step(job, "Required production Vulkan model effects validation")
        self.assertNotIn("${{", native)
        self.assertNotIn("ctest", native)
        with tempfile.TemporaryDirectory(prefix="studiocast-vulkan-models-native-") as directory:
            root = Path(directory)
            env, _ = self.vulkan_models_fixture(root)
            for code in (0, 1, 77):
                result = subprocess.run(["bash", "-c", build], cwd=root, env=env | {"FAKE_BUILD_EXIT": str(code)},
                                        capture_output=True, text=True)
                self.assertEqual(code, result.returncode, result.stderr)
                arguments = (root / "cmake-calls").read_bytes().decode().split("\0")[:-1]
                self.assertEqual(["--build", "build", "--target", "studiocast-vulkan-effects-production-smoke",
                                  "studiocast-vulkan-model-diagnostics-tests"], arguments)
            expected_args = ["--require", "--model-root", str(Path(env["VULKAN_MODEL_ROOT"]).resolve()),
                             "--model-id", env["VULKAN_MODEL_ID"], "--replace-path", env["VULKAN_REPLACE_PATH"],
                             "--face-path", env["VULKAN_FACE_PATH"], "--frames", "3"]
            def invoke(**overrides):
                for name in ("smoke-calls", "diagnostics-env", "smoke-env"):
                    (root / name).unlink(missing_ok=True)
                return subprocess.run(["bash", "-c", native], cwd=root, env=env | overrides,
                                      capture_output=True, text=True)
            for diagnostics, smoke, expected in ((0, 0, 0), (1, 0, 1), (77, 0, 77), (0, 1, 1), (0, 77, 77)):
                with self.subTest(diagnostics=diagnostics, smoke=smoke):
                    result = invoke(FAKE_DIAGNOSTICS_EXIT=str(diagnostics), FAKE_SMOKE_EXIT=str(smoke))
                    self.assertEqual(expected, result.returncode, result.stderr)
                    self.assertTrue((root / "build/qualification/model-diagnostics.log").exists())
                    if diagnostics:
                        self.assertFalse((root / "smoke-calls").exists())
                        continue
                    self.assertEqual(expected_args, (root / "smoke-calls").read_bytes().decode().split("\0")[:-1])
                    cold_env = (root / "diagnostics-env").read_bytes().decode().split("\0")[:-1]
                    smoke_env = (root / "smoke-env").read_bytes().decode().split("\0")[:-1]
                    self.assertEqual(cold_env, smoke_env)
                    self.assertEqual(str(Path(env["VULKAN_MODEL_ROOT"]).parents[2]), cold_env[1])
                    self.assertTrue(Path(cold_env[0]).is_relative_to(Path(env["RUNNER_TEMP"])))
                    isolated = Path(cold_env[0]).parent
                    self.assertEqual([str(isolated / name) for name in ("config", "cache", "state")], cold_env[2:])
                    self.assertNotEqual(env["HOME"], cold_env[0])
                    self.assertFalse((root / "escaped").exists())
                    self.assertTrue((root / "build/qualification/model-effects-production.log").exists())
            self.assertEqual(1, invoke(FAKE_INVENTORY_EXIT="1").returncode)
            self.assertFalse((root / "diagnostics-env").exists())
            missing = [{key: ""} for key in
                       ("VULKAN_MODEL_ROOT", "VULKAN_MODEL_ID", "VULKAN_REPLACE_PATH", "VULKAN_FACE_PATH")]
            missing += [{"VULKAN_MODEL_ROOT": str(root)},
                        {"VULKAN_MODEL_ROOT": str(root / "absent")},
                        {"VULKAN_REPLACE_PATH": str(root / "absent.png")},
                        {"VULKAN_REPLACE_PATH": str(root)},
                        {"VULKAN_FACE_PATH": str(root / "absent.jpg")},
                        {"VULKAN_FACE_PATH": str(root)}]
            for overrides in missing:
                with self.subTest(native_prerequisite=overrides):
                    result = invoke(**overrides)
                    self.assertNotEqual(0, result.returncode)
                    self.assertFalse((root / "diagnostics-env").exists())
                    self.assertFalse((root / "smoke-calls").exists())
            for binary_name in ("studiocast-vulkan-model-diagnostics-tests", "studiocast-vulkan-effects-production-smoke"):
                binary = root / "build" / binary_name
                saved = binary.read_bytes()
                binary.unlink()
                self.assertNotEqual(0, invoke().returncode)
                binary.write_bytes(saved)
                binary.chmod(0o755)

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
