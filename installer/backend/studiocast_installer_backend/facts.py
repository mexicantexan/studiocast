"""Read-only host analysis and normalized installer facts."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .contracts import (
    BackendError,
    CANONICAL_EFFECTS,
    DEFAULT_MODEL_PACKS,
    FACTS_VERSION,
    HELPER_PATH,
    PKEXEC_PATH,
    POLKIT_POLICY_PATH,
    POLKIT_SERVICE_PATH,
    SYSTEM_BUS_PATH,
    V4L_OWNED_PATHS,
    digest,
    file_sha256,
    reject_duplicates,
    roots,
    utc_now,
    version_relation,
)
from .manifest import manifest_state, validated_v4l_ownership
from .packaged import catalog_size_summary, load_model_catalog, privilege_client_module

def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def os_facts() -> dict[str, Any]:
    values = parse_os_release()
    os_id, version = values.get("ID", "unknown"), values.get("VERSION_ID", "")
    codename = values.get("UBUNTU_CODENAME") or values.get("VERSION_CODENAME", "")
    base_id, base_version = os_id, version
    if os_id == "linuxmint" or "ubuntu" in values.get("ID_LIKE", "").split():
        base_id = "ubuntu"
        base_version = {"jammy": "22.04", "noble": "24.04"}.get(codename, "")
        if os_id == "linuxmint" and not base_version:
            base_version = "22.04" if version.startswith("21") else "24.04" if version.startswith("22") else ""
    supported = base_id == "ubuntu" and base_version in {"22.04", "24.04"} and platform.machine() in {"x86_64", "amd64"}
    code = ("os.supported.ubuntu_jammy" if base_version == "22.04" else
            "os.supported.ubuntu_noble" if base_version == "24.04" else
            "arch.unsupported.machine" if platform.machine() not in {"x86_64", "amd64"} else
            "os.unsupported.distribution")
    return {"id": os_id, "version_id": version, "base_id": base_id,
            "base_version_id": base_version, "architecture": platform.machine(),
            "kernel_release": platform.release(), "supported": supported, "reason_codes": [code]}

def helper_facts() -> dict[str, Any]:
    unavailable = {"path": str(HELPER_PATH), "present": False, "trusted": False,
        "compatible": False, "authorization_transport_available": False,
        "helper_version": None, "protocol_versions": [], "pkexec_path": str(PKEXEC_PATH),
        "policy_path": str(POLKIT_POLICY_PATH), "reason_codes": ["privileged_helper_missing"]}
    test_mode = os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1"
    client = privilege_client_module()
    complete_test_contract = all(os.environ.get(name) for name in (
        "STUDIOCAST_INSTALLER_TEST_HELPER", "STUDIOCAST_INSTALLER_TEST_PKEXEC",
        "STUDIOCAST_INSTALLER_TEST_POLKIT_POLICY", "STUDIOCAST_INSTALLER_TEST_SYSTEM_BUS",
        "STUDIOCAST_INSTALLER_TEST_POLKIT_SERVICE"))
    if client is not None and (not test_mode or complete_test_contract):
        kwargs: dict[str, Any] = {}
        if test_mode and os.environ.get("STUDIOCAST_INSTALLER_TEST_HELPER"):
            kwargs = {
                "helper_path": Path(os.environ["STUDIOCAST_INSTALLER_TEST_HELPER"]),
                "pkexec_path": Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_PKEXEC", str(PKEXEC_PATH))),
                "policy_path": Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_POLKIT_POLICY", str(POLKIT_POLICY_PATH))),
                "system_bus_path": Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_SYSTEM_BUS", str(SYSTEM_BUS_PATH))),
                "polkit_service_path": Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_POLKIT_SERVICE", str(POLKIT_SERVICE_PATH))),
                "required_uid": os.getuid(), "test_mode": True,
                "version_probe": lambda _path: json.dumps(
                    {"helper_version": "1.0.0", "protocol_versions": [1]}),
            }
        try:
            return client.preflight_trusted_helper(**kwargs).as_dict()
        except Exception:
            return {**unavailable, "reason_codes": ["privileged_helper_untrusted"]}
    if not test_mode:
        # Packaging must include the read-only trusted-client contract.  A mode
        # check alone is not enough to establish helper compatibility, the
        # exact polkit binding, or authorization transport availability.
        return {**unavailable, "reason_codes": ["privileged_helper_incompatible"]}
    # Compatibility for hermetic backend tests on a branch before the packaged
    # client module is staged.  This path is unreachable in production.
    helper = Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_HELPER", ""))
    pkexec = Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_PKEXEC", ""))
    try:
        trusted = all(path.is_file() and not path.is_symlink() and
                      path.stat().st_uid == os.getuid() and
                      not (path.stat().st_mode & 0o022) and os.access(path, os.X_OK)
                      for path in (helper, pkexec))
    except OSError:
        trusted = False
    return {**unavailable, "path": str(helper), "present": helper.is_file(),
        "trusted": trusted, "compatible": trusted,
        "authorization_transport_available": trusted, "helper_version": "test",
        "protocol_versions": [1] if trusted else [], "pkexec_path": str(pkexec),
        "reason_codes": ["privileged_helper_usable" if trusted else "privileged_helper_missing"]}


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def readonly_command(argv: list[str], timeout: float = 3.0) -> tuple[int | None, str]:
    """Run a bounded, read-only capability probe without a shell."""
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout,
                                check=False, errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    return result.returncode, result.stdout[:4 * 1024 * 1024]


def writable_path_fact(path: Path) -> dict[str, Any]:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    writable = current.is_dir() and not current.is_symlink() and os.access(current, os.W_OK | os.X_OK)
    return {"path": str(path), "writable": writable,
            "reason_codes": ["path.writable" if writable else "path.not_writable"]}


def nvidia_facts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reason_codes: list[str] = []
    devices: list[dict[str, Any]] = []
    if not command_exists("nvidia-smi"):
        return devices, {"driver_usable": False, "cuda_usable": False,
                         "reason_codes": ["gpu.nvidia.tool_unavailable"]}
    code, text = readonly_command([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return devices, {"driver_usable": False, "cuda_usable": False,
                         "reason_codes": ["gpu.nvidia.driver_unusable"]}
    for row in csv.reader(text.splitlines()):
        if len(row) != 5:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        devices.append({"id": f"nvidia:{index}", "index": index,
                        "uuid": row[1].strip(), "name": row[2].strip(),
                        "vendor": "nvidia", "driver_version": row[3].strip(),
                        "compute_capability": row[4].strip()})
    driver_usable = bool(devices)
    summary_code, summary = readonly_command(["nvidia-smi"], timeout=3)
    cuda_match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", summary)
    cuda_usable = driver_usable and summary_code == 0 and cuda_match is not None
    reason_codes.append("gpu.nvidia.driver_usable" if driver_usable else "gpu.nvidia.no_devices")
    reason_codes.append("gpu.cuda.driver_api_reported" if cuda_usable else "gpu.cuda.runtime_unavailable")
    return devices, {"driver_usable": driver_usable, "cuda_usable": cuda_usable,
                     "reported_cuda_version": cuda_match.group(1) if cuda_match else None,
                     "reason_codes": reason_codes}


def vulkan_facts() -> dict[str, Any]:
    loader = command_exists("vulkaninfo") or any(Path(path).exists() for path in (
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/lib/x86_64-linux-gnu/libvulkan.so.1"))
    result = {"loader_present": loader, "physical_devices": [],
              "compute_device_usable": False, "reason_codes": []}
    if not loader:
        result["reason_codes"] = ["vulkan_loader_unavailable"]
        return result
    if not command_exists("vulkaninfo"):
        result["reason_codes"] = ["vulkan_no_physical_device", "vulkan_probe_tool_unavailable"]
        return result
    summary_code, summary = readonly_command(["vulkaninfo", "--summary"], timeout=5)
    full_code, full = readonly_command(["vulkaninfo"], timeout=8)
    if summary_code != 0:
        result["reason_codes"] = ["vulkan_no_physical_device", "vulkan_probe_failed"]
        return result
    devices: list[dict[str, Any]] = []
    blocks = re.split(r"(?m)^GPU(?=\d+:)", summary)
    for block in blocks[1:]:
        index_match = re.match(r"(\d+):", block)
        name_match = re.search(r"(?m)^\s*deviceName\s*=\s*(.+?)\s*$", block)
        type_match = re.search(r"(?m)^\s*deviceType\s*=\s*(\S+)\s*$", block)
        if not index_match:
            continue
        index = int(index_match.group(1))
        device_type = type_match.group(1) if type_match else "unknown"
        full_block_match = re.search(
            rf"(?ms)^GPU id = {index}(?: \([^\n]*\))?.*?(?=^GPU id = \d+|\Z)", full)
        full_block = full_block_match.group(0) if full_block_match else ""
        compute = full_code == 0 and ("VK_QUEUE_COMPUTE_BIT" in full_block or
                                      re.search(r"queueFlags\s*=.*\bCOMPUTE\b", full_block) is not None)
        software = "CPU" in device_type.upper()
        devices.append({"index": index, "name": name_match.group(1).strip() if name_match else "",
                        "device_type": device_type, "compute_queue": compute,
                        "software_device": software, "production_compute_usable": compute and not software})
    result["physical_devices"] = devices
    result["compute_device_usable"] = any(item["production_compute_usable"] for item in devices)
    if not devices:
        result["reason_codes"] = ["vulkan_no_physical_device"]
    elif result["compute_device_usable"]:
        result["reason_codes"] = ["vulkan_compute_device_usable"]
    elif all(item["software_device"] for item in devices):
        result["reason_codes"] = ["vulkan_only_cpu_devices_available"]
    else:
        result["reason_codes"] = ["vulkan_no_compute_queue"]
    return result


def runtime_diagnostics(data_root: Path) -> dict[str, Any] | None:
    ctl = data_root / "current/bin/studiocastctl"
    if not ctl.is_file() or ctl.is_symlink() or not os.access(ctl, os.X_OK):
        return None

    # GET_STATUS is deliberately cache-only.  A newly started daemon has not
    # populated its expensive provider/device diagnostics yet, so refresh them
    # through the installed public CLI before consuming status.  The CLI owns
    # the daemon protocol and bounds the IPC call at 30 seconds; this outer
    # timeout also bounds process startup/teardown.  Older CLIs/daemons fail
    # this additive command without preventing the compatibility status read,
    # but their unrefreshed evidence is never eligible for recommendation.
    refresh_code, refresh_text = readonly_command(
        [str(ctl), "diagnostics", "refresh"], timeout=35)
    code, text = readonly_command([str(ctl), "status"], timeout=4)
    if code != 0:
        return None
    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
        refreshed = json.loads(refresh_text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, BackendError):
        return None
    refresh_engines = refreshed.get("engines") if isinstance(refreshed, dict) else None
    refresh_open_vulkan = (refresh_engines.get("open_vulkan")
                           if isinstance(refresh_engines, dict) else None)
    if (refresh_code != 0 or not isinstance(refresh_engines, dict) or
            not isinstance(refresh_open_vulkan, dict) or not refresh_open_vulkan or
            not isinstance(value, dict)):
        return None
    return value


def ort_facts(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    engines = diagnostics.get("engines", {}) if diagnostics else {}
    candidates = [engines.get("open_cuda", {}), engines.get("open_audio", {})]
    candidates = [value for value in candidates if isinstance(value, dict)]
    version = next((str(value.get("onnxruntime_version")) for value in candidates
                    if value.get("onnxruntime_version")), None)
    provider_names = sorted({str(provider) for value in candidates
                             for provider in value.get("onnxruntime_providers", [])
                             if isinstance(provider, str)})
    if candidates and (version or provider_names):
        cpu = any(bool(value.get("onnxruntime_cpu_provider_present")) for value in candidates)
        cuda = any(bool(value.get("onnxruntime_cuda_provider_present")) and
                   bool(value.get("cuda_context_available", True)) for value in candidates)
        return {"present": True, "compatible": True, "version": version,
                "provider_names": provider_names, "providers": {"cpu": cpu, "cuda": cuda},
                "probe": "daemon_runtime_diagnostics",
                "reason_codes": ["runtime.ort.providers_usable"]}
    code, version_text = readonly_command(["pkg-config", "--modversion", "onnxruntime"])
    present = code == 0 and bool(version_text.strip())
    return {"present": present, "compatible": present,
            "version": version_text.strip() if present else None,
            "provider_names": [], "providers": {"cpu": False, "cuda": False},
            "probe": "metadata_only" if present else "unavailable",
            "reason_codes": ["runtime.ort.present_providers_unvalidated" if present
                             else "runtime.ort.unavailable"]}


def maxine_facts(diagnostics: dict[str, Any] | None, data_root: Path) -> dict[str, Any]:
    value = diagnostics.get("engines", {}).get("maxine", {}) if diagnostics else {}
    if isinstance(value, dict) and value:
        components: dict[str, Any] = {}
        for source, target in (("vfx", "video_fx"), ("ar", "ar"), ("afx", "audio_fx")):
            component = value.get(source, {})
            components[target] = {"present": bool(component.get("root_found")),
                                  "library_loadable": bool(component.get("library_loadable")),
                                  "usable": bool(component.get("ok"))}
        effects = {effect: {"usable": effect in value.get("available_effects", []),
                            "reason_code": "maxine.effect.usable" if effect in value.get("available_effects", [])
                            else "maxine.effect.unavailable"} for effect in CANONICAL_EFFECTS}
        return {"components": components, "effects": effects,
                "usable": bool(value.get("ok")), "probe": "daemon_runtime_diagnostics",
                "reason_codes": ["maxine.sdk.usable" if value.get("ok") else
                                 str(value.get("blocked_reason") or "maxine.sdk.unusable")]}
    roots_by_component = {"video_fx": data_root / "maxine/VideoFX",
                          "ar": data_root / "maxine/ARSDK",
                          "audio_fx": data_root / "maxine/Audio_Effects_SDK"}
    components = {name: {"present": path.is_dir() and not path.is_symlink(),
                         "library_loadable": False, "usable": False}
                  for name, path in roots_by_component.items()}
    present = any(item["present"] for item in components.values())
    return {"components": components, "effects": {}, "usable": False,
            "probe": "layout_only", "reason_codes": ["maxine.sdk.present_unvalidated" if present
                                                       else "maxine.sdk.absent"]}


def common_vulkan_effect_evidence(engine: Any) -> tuple[dict[str, Any], str | None]:
    diagnostics_available = isinstance(engine, dict) and bool(engine)
    value = engine if diagnostics_available else {}
    evidence = {
        "runtime_diagnostics_available": diagnostics_available,
        "backend_compiled": value.get("compiled_enabled") is True,
        "runtime_library_found": value.get("runtime_library_found") is True,
        "instance_created": value.get("instance_created") is True,
        "physical_device_found": value.get("physical_device_found") is True,
        "non_cpu_device_selected": value.get("non_cpu_device_selected") is True,
        "compute_queue_available": value.get("compute_queue_available") is True,
        "logical_device_created": value.get("logical_device_created") is True,
        "context_created": value.get("context_created") is True,
        "context_healthy": value.get("context_healthy") is True,
        "production_hardware_ready": value.get("production_hardware_ready") is True,
        "utility_kernels_ready": (value.get("shader_pipeline_created") is True and
                                  value.get("ok") is True),
    }
    evidence["common_ready"] = all(evidence.values())
    if not diagnostics_available:
        return evidence, "open_vulkan_runtime_diagnostics_unavailable"
    if not evidence["backend_compiled"]:
        return evidence, "vulkan_backend_disabled_in_build"
    if not evidence["runtime_library_found"]:
        return evidence, "vulkan_runtime_not_found"
    if not evidence["instance_created"]:
        return evidence, str(value.get("fallback_reason") or
                             "vulkan_instance_create_failed")
    if not evidence["physical_device_found"]:
        reason = str(value.get("fallback_reason") or "vulkan_no_physical_device")
        return evidence, reason
    if value.get("cpu_device_selected") is True or not evidence["non_cpu_device_selected"]:
        return evidence, "vulkan_only_cpu_devices_available"
    if not evidence["compute_queue_available"]:
        reason = str(value.get("fallback_reason") or "vulkan_no_compute_queue")
        return evidence, reason
    if not evidence["logical_device_created"]:
        return evidence, "vulkan_device_create_failed"
    if not evidence["context_created"] or not evidence["context_healthy"]:
        return evidence, str(value.get("context_failure_reason") or
                             "vulkan_context_uninitialized")
    if not evidence["production_hardware_ready"]:
        return evidence, "vulkan_production_hardware_not_ready"
    if not evidence["utility_kernels_ready"]:
        return evidence, "open_vulkan_utility_kernels_unavailable"
    return evidence, None


def effect_capabilities(diagnostics: dict[str, Any] | None, maxine: dict[str, Any]) -> dict[str, Any]:
    engines = diagnostics.get("engines", {}) if diagnostics else {}
    open_cuda = engines.get("open_cuda", {})
    open_cuda = open_cuda if isinstance(open_cuda, dict) else {}
    open_vulkan = engines.get("open_vulkan", {})
    open_vulkan = open_vulkan if isinstance(open_vulkan, dict) else {}
    cuda_available = set(open_cuda.get("available_effects", []))
    cuda_blocked = open_cuda.get("blocked_effects", {})
    cuda_blocked = cuda_blocked if isinstance(cuda_blocked, dict) else {}
    vulkan_available = set(open_vulkan.get("available_effects", []))
    vulkan_common, vulkan_common_blocker = common_vulkan_effect_evidence(open_vulkan)

    out: dict[str, Any] = {}
    for effect in CANONICAL_EFFECTS:
        maxine_effect = maxine.get("effects", {}).get(effect, {})
        maxine_usable = isinstance(maxine_effect, dict) and maxine_effect.get("usable") is True
        cuda_usable = effect in cuda_available
        success_codes: dict[str, list[str]] = {"maxine": [], "cuda": [], "vulkan": [], "cpu": []}
        blocker_codes: dict[str, list[str]] = {"maxine": [], "cuda": [], "vulkan": [], "cpu": []}
        if maxine_usable:
            success_codes["maxine"] = [str(maxine_effect.get("reason_code") or
                                             "maxine.effect.usable")]
        else:
            blocker_codes["maxine"] = [str(maxine_effect.get("reason_code") or
                                             "maxine.effect.unavailable")]
        if cuda_usable:
            success_codes["cuda"] = ["open_cuda_effect_available"]
        else:
            blocker_codes["cuda"] = [str(cuda_blocked.get(effect) or
                                           "open_cuda_effect_unavailable")]
        blocker_codes["cpu"] = [f"effect.{effect}.cpu.no_selectable_production_path"]

        evidence = copy.deepcopy(vulkan_common)
        evidence.update({
            "live_stage_implemented": False,
            "runtime_ready": False,
            "model_artifact_ready": False,
            "warmup_complete": False,
            "exact_device_and_queue": False,
            "device_resident": False,
            "synchronization_contract_validated": False,
            "bounded_reusable_allocations": False,
            "parity_validated": False,
            "no_cpu_layers": False,
            "no_dynamic_allocations": False,
            "no_cpu_readback": False,
            "no_cpu_fallback": False,
            "temporal_history_device_resident": False,
            "temporal_history_bounded": False,
            "history_reset_on_disable": False,
            "history_reset_on_reconfigure": False,
            "capture_sequence_discontinuity_reset": False,
            "cpu_tail": False,
            "selectable_cpu_fallback": False,
            "parameter_contract": None,
            "daemon_production_ready": False,
            "effect_production_ready_attested": False,
            "production_ready": False,
            "degraded_reason_codes": [],
        })
        vulkan_status = "unavailable"
        specific_blocker: str | None = None

        if effect == "mirror":
            evidence["daemon_production_ready"] = (
                open_vulkan.get("mirror_production_ready") is True)
            evidence["live_stage_implemented"] = True
            evidence["runtime_ready"] = True
            evidence["model_artifact_ready"] = True
            evidence["warmup_complete"] = True
            evidence["exact_device_and_queue"] = evidence["common_ready"]
            evidence["device_resident"] = True
            evidence["synchronization_contract_validated"] = True
            evidence["bounded_reusable_allocations"] = True
            evidence["parity_validated"] = True
            evidence["no_cpu_layers"] = True
            evidence["no_dynamic_allocations"] = True
            evidence["no_cpu_readback"] = True
            evidence["no_cpu_fallback"] = True
            exact_ready = (open_vulkan.get("mirror_production_ready") is True and
                           open_vulkan.get("mirror_readiness_code") ==
                           "open_vulkan_mirror_production_ready")
            evidence["effect_production_ready_attested"] = exact_ready
            evidence["production_ready"] = bool(
                evidence["common_ready"] and exact_ready and effect in vulkan_available)
            if evidence["production_ready"]:
                vulkan_status = "production_usable"
                success_codes["vulkan"] = ["open_vulkan_mirror_production_ready"]
            elif vulkan_common_blocker:
                specific_blocker = vulkan_common_blocker
            elif "mirror_production_ready" not in open_vulkan:
                specific_blocker = "open_vulkan_mirror_production_evidence_missing"
            elif open_vulkan.get("mirror_production_ready") is not True:
                specific_blocker = str(open_vulkan.get("mirror_blocker_code") or
                                       "open_vulkan_mirror_unavailable")
            else:
                specific_blocker = "open_vulkan_mirror_live_stage_unavailable"
        elif effect == "vignette":
            evidence["daemon_production_ready"] = (
                open_vulkan.get("vignette_fixed_center_production_ready") is True)
            evidence["live_stage_implemented"] = True
            evidence["runtime_ready"] = True
            evidence["model_artifact_ready"] = True
            evidence["warmup_complete"] = True
            evidence["exact_device_and_queue"] = evidence["common_ready"]
            evidence["device_resident"] = True
            evidence["synchronization_contract_validated"] = True
            evidence["bounded_reusable_allocations"] = True
            evidence["parity_validated"] = True
            evidence["no_cpu_layers"] = True
            evidence["no_dynamic_allocations"] = True
            evidence["no_cpu_readback"] = True
            evidence["no_cpu_fallback"] = True
            evidence["parameter_contract"] = open_vulkan.get("vignette_parameter_contract")
            exact_ready = (
                open_vulkan.get("vignette_fixed_center_production_ready") is True and
                open_vulkan.get("vignette_readiness_code") ==
                "open_vulkan_vignette_fixed_center_production_ready" and
                evidence["parameter_contract"] == "fixed_center")
            evidence["effect_production_ready_attested"] = exact_ready
            evidence["production_ready"] = bool(
                evidence["common_ready"] and exact_ready and effect in vulkan_available)
            if evidence["production_ready"]:
                vulkan_status = "production_usable"
                success_codes["vulkan"] = [
                    "open_vulkan_vignette_fixed_center_production_ready"]
            elif vulkan_common_blocker:
                specific_blocker = vulkan_common_blocker
            elif evidence["parameter_contract"] != "fixed_center":
                specific_blocker = "vulkan_vignette_tracked_center_not_supported"
            elif "vignette_fixed_center_production_ready" not in open_vulkan:
                specific_blocker = "open_vulkan_vignette_production_evidence_missing"
            elif open_vulkan.get("vignette_fixed_center_production_ready") is not True:
                specific_blocker = str(open_vulkan.get("vignette_blocker_code") or
                                       "open_vulkan_vignette_unavailable")
            else:
                specific_blocker = "open_vulkan_vignette_live_stage_unavailable"
        elif effect == "auto_frame":
            evidence["daemon_production_ready"] = (
                open_vulkan.get("auto_frame_production_ready") is True)
            evidence["live_stage_implemented"] = (
                open_vulkan.get("auto_frame_crop_stage_implemented") is True)
            evidence["cpu_tail"] = open_vulkan.get("auto_frame_cpu_tail") is True
            evidence["degraded_reason_codes"] = [str(
                open_vulkan.get("auto_frame_degraded_reason_code") or
                "vulkan_effect_cpu_tail")]
            evidence["selectable_cpu_fallback"] = (
                open_vulkan.get("auto_frame_selectable_cpu_fallback") is True)
            # The real crop stage and its CPU tracking tail are implementation
            # facts, not production-ready or selectable CPU-engine evidence.
            evidence["production_ready"] = False
            if (effect in vulkan_available and evidence["live_stage_implemented"] and
                    evidence["cpu_tail"]):
                vulkan_status = "usable_with_degraded_behavior"
            specific_blocker = (vulkan_common_blocker or str(
                open_vulkan.get("auto_frame_blocker_code") or
                "vulkan_auto_frame_yunet_unavailable"))
        elif effect in {"virtual_background.blur", "virtual_background.remove",
                        "virtual_background.replace", "virtual_key_light"}:
            evidence["daemon_production_ready"] = (
                open_vulkan.get("matting_production_ready") is True)
            evidence["live_stage_implemented"] = True
            evidence["runtime_ready"] = (open_vulkan.get("matting_runtime_created") is True and
                                         open_vulkan.get("matting_graph_loaded") is True)
            evidence["model_artifact_ready"] = (
                open_vulkan.get("matting_model_pack_selected") is True and
                open_vulkan.get("matting_model_contract_validated") is True)
            evidence["warmup_complete"] = open_vulkan.get("matting_warmup_complete") is True
            evidence["exact_device_and_queue"] = (
                open_vulkan.get("matting_shared_device_imported") is True and
                open_vulkan.get("matting_queue_ownership_explicit") is True)
            evidence["device_resident"] = (
                open_vulkan.get("input_device_resident") is True and
                open_vulkan.get("alpha_device_resident") is True and
                open_vulkan.get("output_device_resident") is True and
                open_vulkan.get("device_residency_mode") == "device_resident" and
                open_vulkan.get("matting_cpu_layers_used") is False and
                open_vulkan.get("matting_cpu_readback_count") == 0)
            evidence["synchronization_contract_validated"] = (
                open_vulkan.get("matting_synchronous_completion") is True)
            evidence["bounded_reusable_allocations"] = (
                open_vulkan.get("matting_bounded_reusable_allocations") is True)
            evidence["parity_validated"] = (
                open_vulkan.get("matting_parity_validated") is True)
            evidence["no_cpu_layers"] = (
                open_vulkan.get("matting_cpu_layers_used") is False)
            evidence["no_dynamic_allocations"] = (
                open_vulkan.get("matting_dynamic_allocation_count") == 0)
            evidence["no_cpu_readback"] = (
                open_vulkan.get("matting_cpu_readback_count") == 0)
            evidence["no_cpu_fallback"] = (
                open_vulkan.get("matting_cpu_fallback_count") == 0)
            matting_ready = (
                open_vulkan.get("matting_production_ready") is True and
                open_vulkan.get("matting_build_enabled") is True and
                open_vulkan.get("matting_adapter_available") is True and
                evidence["runtime_ready"] and evidence["model_artifact_ready"] and
                evidence["warmup_complete"] and evidence["exact_device_and_queue"] and
                evidence["device_resident"] and
                evidence["synchronization_contract_validated"] and
                evidence["bounded_reusable_allocations"] and
                evidence["parity_validated"] and evidence["no_cpu_layers"] and
                evidence["no_dynamic_allocations"] and
                evidence["no_cpu_readback"] and evidence["no_cpu_fallback"])
            evidence["effect_production_ready_attested"] = matting_ready
            evidence["production_ready"] = bool(
                evidence["common_ready"] and matting_ready and effect in vulkan_available)
            if evidence["production_ready"]:
                vulkan_status = "production_usable"
                success_codes["vulkan"] = ["open_vulkan_matting_production_ready"]
            else:
                vulkan_status = "experimental"
                specific_blocker = (vulkan_common_blocker or str(
                    open_vulkan.get("matting_reason_code") or
                    "open_vulkan_matting_unavailable"))
                nested = open_vulkan.get("matting_blocker_code")
                if nested and nested != specific_blocker:
                    blocker_codes["vulkan"].append(str(nested))
        elif effect == "eye_contact":
            evidence["daemon_production_ready"] = (
                open_vulkan.get("eye_contact_production_ready") is True)
            evidence["live_stage_implemented"] = (
                open_vulkan.get("eye_contact_live_stage_implemented") is True)
            evidence["runtime_ready"] = (
                open_vulkan.get("eye_contact_production_adapter_available") is True and
                open_vulkan.get("eye_contact_vulkan_inference_provider_available") is True)
            evidence["model_artifact_ready"] = (
                open_vulkan.get("eye_contact_model_pack_selected") is True and
                open_vulkan.get("eye_contact_artifact_contract_validated") is True)
            evidence["warmup_complete"] = open_vulkan.get("eye_contact_warmup_complete") is True
            evidence["exact_device_and_queue"] = (
                open_vulkan.get("eye_contact_shared_device_imported") is True and
                open_vulkan.get("eye_contact_queue_ownership_explicit") is True)
            evidence["device_resident"] = (
                open_vulkan.get("eye_contact_device_resident_analysis") is True and
                open_vulkan.get("eye_contact_device_resident_tensor_io") is True)
            evidence["selectable_cpu_fallback"] = (
                open_vulkan.get("eye_contact_selectable_cpu_fallback") is True)
            evidence["synchronization_contract_validated"] = (
                open_vulkan.get("eye_contact_synchronization_contract_validated") is True)
            evidence["bounded_reusable_allocations"] = (
                open_vulkan.get("eye_contact_bounded_reusable_allocations") is True)
            evidence["parity_validated"] = (
                open_vulkan.get("eye_contact_parity_validated") is True)
            evidence["no_cpu_readback"] = (
                open_vulkan.get("eye_contact_cpu_readback_count") == 0)
            evidence["no_cpu_fallback"] = (
                open_vulkan.get("eye_contact_cpu_fallback_count") == 0)
            evidence["production_ready"] = bool(
                evidence["common_ready"] and
                open_vulkan.get("eye_contact_production_ready") is True and
                evidence["live_stage_implemented"] and evidence["runtime_ready"] and
                evidence["model_artifact_ready"] and evidence["warmup_complete"] and
                evidence["exact_device_and_queue"] and evidence["device_resident"] and
                evidence["bounded_reusable_allocations"] and
                evidence["synchronization_contract_validated"] and
                evidence["parity_validated"] and evidence["no_cpu_readback"] and
                evidence["no_cpu_fallback"] and
                effect in vulkan_available)
            evidence["effect_production_ready_attested"] = bool(
                open_vulkan.get("eye_contact_production_ready") is True)
            if evidence["production_ready"]:
                vulkan_status = "production_usable"
                success_codes["vulkan"] = ["open_vulkan_eye_contact_production_ready"]
            else:
                specific_blocker = (vulkan_common_blocker or str(
                    open_vulkan.get("eye_contact_reason_code") or
                    "open_vulkan_eye_contact_unavailable"))
                nested = open_vulkan.get("eye_contact_blocker_code")
                if nested and nested != specific_blocker:
                    blocker_codes["vulkan"].append(str(nested))
        elif effect == "video_noise_removal":
            evidence["daemon_production_ready"] = (
                open_vulkan.get("video_noise_removal_production_ready") is True)
            evidence["live_stage_implemented"] = (
                open_vulkan.get("video_noise_removal_live_stage_implemented") is True)
            evidence["runtime_ready"] = (
                open_vulkan.get("video_noise_removal_production_adapter_available") is True and
                open_vulkan.get("video_noise_removal_vulkan_inference_provider_available") is True)
            evidence["model_artifact_ready"] = (
                open_vulkan.get("video_noise_removal_model_pack_selected") is True and
                open_vulkan.get("video_noise_removal_artifact_contract_validated") is True)
            evidence["warmup_complete"] = (
                open_vulkan.get("video_noise_removal_warmup_complete") is True)
            evidence["exact_device_and_queue"] = (
                open_vulkan.get("video_noise_removal_shared_device_imported") is True and
                open_vulkan.get("video_noise_removal_queue_ownership_explicit") is True)
            evidence["device_resident"] = (
                open_vulkan.get("video_noise_removal_fully_device_resident_tensor_io") is True and
                open_vulkan.get("video_noise_removal_device_resident_preprocess") is True and
                open_vulkan.get("video_noise_removal_device_resident_postprocess") is True and
                open_vulkan.get("video_noise_removal_temporal_history_device_resident") is True)
            evidence["selectable_cpu_fallback"] = (
                open_vulkan.get("video_noise_removal_selectable_cpu_fallback") is True)
            evidence["synchronization_contract_validated"] = (
                open_vulkan.get(
                    "video_noise_removal_synchronization_contract_validated") is True)
            evidence["bounded_reusable_allocations"] = (
                open_vulkan.get(
                    "video_noise_removal_bounded_reusable_allocations") is True)
            evidence["parity_validated"] = (
                open_vulkan.get("video_noise_removal_parity_validated") is True)
            evidence["no_cpu_readback"] = (
                open_vulkan.get("video_noise_removal_cpu_readback_count") == 0)
            evidence["no_cpu_fallback"] = (
                open_vulkan.get("video_noise_removal_cpu_fallback_count") == 0)
            evidence["temporal_history_device_resident"] = (
                open_vulkan.get(
                    "video_noise_removal_temporal_history_device_resident") is True)
            evidence["temporal_history_bounded"] = (
                open_vulkan.get("video_noise_removal_temporal_history_bounded") is True)
            evidence["history_reset_on_disable"] = (
                open_vulkan.get("video_noise_removal_history_reset_on_disable") is True)
            evidence["history_reset_on_reconfigure"] = (
                open_vulkan.get(
                    "video_noise_removal_history_reset_on_reconfigure") is True)
            evidence["capture_sequence_discontinuity_reset"] = (
                open_vulkan.get(
                    "video_noise_removal_capture_sequence_discontinuity_reset") is True)
            evidence["production_ready"] = bool(
                evidence["common_ready"] and
                open_vulkan.get("video_noise_removal_production_ready") is True and
                evidence["live_stage_implemented"] and evidence["runtime_ready"] and
                evidence["model_artifact_ready"] and evidence["warmup_complete"] and
                evidence["exact_device_and_queue"] and evidence["device_resident"] and
                evidence["synchronization_contract_validated"] and
                evidence["bounded_reusable_allocations"] and
                evidence["temporal_history_device_resident"] and
                evidence["temporal_history_bounded"] and
                evidence["history_reset_on_disable"] and
                evidence["history_reset_on_reconfigure"] and
                evidence["capture_sequence_discontinuity_reset"] and
                evidence["parity_validated"] and evidence["no_cpu_readback"] and
                evidence["no_cpu_fallback"] and
                effect in vulkan_available)
            evidence["effect_production_ready_attested"] = bool(
                open_vulkan.get("video_noise_removal_production_ready") is True)
            if evidence["production_ready"]:
                vulkan_status = "production_usable"
                success_codes["vulkan"] = [
                    "open_vulkan_video_noise_removal_production_ready"]
            else:
                specific_blocker = (vulkan_common_blocker or str(
                    open_vulkan.get("video_noise_removal_reason_code") or
                    "open_vulkan_video_noise_removal_unavailable"))
                nested = open_vulkan.get("video_noise_removal_blocker_code")
                if nested and nested != specific_blocker:
                    blocker_codes["vulkan"].append(str(nested))

        if specific_blocker:
            blocker_codes["vulkan"].insert(0, specific_blocker)
        out[effect] = {
            "maxine": "production_usable" if maxine_usable else "unavailable",
            "cuda": "production_usable" if cuda_usable else "unavailable",
            "vulkan": vulkan_status,
            "cpu": "unavailable",
            "success_reason_codes": success_codes,
            "blocker_reason_codes": blocker_codes,
            "vulkan_evidence": evidence,
            "cpu_evidence": {
                "selectable_production_path": False,
                "reason_code": f"effect.{effect}.cpu.no_selectable_production_path",
            },
        }
    return out
def host_analyze(target_version: str | None) -> dict[str, Any]:
    r = roots()
    classification, manifest, install_reasons = manifest_state(r["data"] / "install-manifest.json")
    active = None
    previous = None
    desired = None
    if manifest:
        if manifest.get("schema_version") == 2:
            active = manifest.get("versions", {}).get("active")
            previous = manifest.get("versions", {}).get("previous")
            desired = manifest.get("desired_configuration")
        else:
            active = manifest.get("installed_version")
    try:
        disk = shutil.disk_usage(r["cache"]).free
    except OSError:
        try:
            disk = shutil.disk_usage(r["home"]).free
        except OSError:
            disk = 0
    gpu_devices: list[dict[str, Any]] = []
    try:
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            vendor = (card / "device/vendor").read_text().strip()
            device = (card / "device/device").read_text().strip()
            vendor_name = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}.get(vendor.lower(), "unknown")
            gpu_devices.append({"id": card.name, "vendor_id": vendor, "device_id": device,
                                "vendor": vendor_name})
    except OSError:
        pass
    nvidia_devices, nvidia = nvidia_facts()
    known_ids = {item["id"] for item in gpu_devices}
    gpu_devices.extend(item for item in nvidia_devices if item["id"] not in known_ids)
    vulkan = vulkan_facts()
    module_loaded = Path("/sys/module/v4l2loopback").exists()
    modinfo_code, modinfo_path = readonly_command(["modinfo", "-n", "v4l2loopback"], timeout=2)
    module_available = module_loaded or (modinfo_code == 0 and bool(modinfo_path.strip()))
    device_present = Path("/dev/video10").exists()
    device_label = ""
    try:
        device_label = Path("/sys/class/video4linux/video10/name").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    device_conflict = device_present and bool(device_label) and "studiocast" not in device_label.lower()
    persistence_paths = (Path("/etc/modules-load.d/studiocast-v4l2loopback.conf"),
                         Path("/etc/modprobe.d/studiocast-v4l2loopback.conf"))
    persistence_state = "configured" if all(path.is_file() and not path.is_symlink() for path in persistence_paths) else "partial" if any(path.exists() for path in persistence_paths) else "absent"
    package_state = "unknown"
    if command_exists("dpkg-query"):
        try:
            package_probe = subprocess.run(["dpkg-query", "-W", "-f=${Status}", "v4l2loopback-utils"],
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           text=True, timeout=2, check=False)
            package_state = "installed" if package_probe.returncode == 0 and "install ok installed" in package_probe.stdout else "missing"
        except (OSError, subprocess.TimeoutExpired):
            pass
    headers = Path(f"/lib/modules/{platform.release()}/build").exists()
    dkms_present = command_exists("dkms")
    secure_boot = "unknown"
    try:
        values = list(Path("/sys/firmware/efi/efivars").glob("SecureBoot-*"))
        if values:
            secure_boot = "enabled" if values[0].read_bytes()[-1:] == b"\x01" else "disabled"
    except OSError:
        pass
    systemd_present = command_exists("systemctl")
    systemd_usable = False
    systemd_reason = "service.user_manager.unavailable"
    if systemd_present:
        try:
            probe = subprocess.run(["systemctl", "--user", "show-environment"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=2, check=False)
            systemd_usable = probe.returncode == 0
            systemd_reason = "service.user_manager.usable" if systemd_usable else "service.user_manager.unreachable"
        except (OSError, subprocess.TimeoutExpired):
            systemd_reason = "service.user_manager.unreachable"
    diagnostics = runtime_diagnostics(r["data"])
    ort = ort_facts(diagnostics)
    maxine = maxine_facts(diagnostics, r["data"])
    effects = effect_capabilities(diagnostics, maxine)
    owned_v4l = validated_v4l_ownership(manifest)
    model_packs: dict[str, Any] = {}
    _, catalog, _, catalog_hash = load_model_catalog()
    for pack in catalog["packs"]:
        pack_id = pack["id"]
        root = r["data"] / "models" / pack["family"] / pack["pack_path"]
        artifacts = pack["artifacts"]
        present = all((root / artifact["name"]).is_file() for artifact in artifacts)
        verified = all((root / artifact["name"]).is_file() and
                       not (root / artifact["name"]).is_symlink() and
                       (root / artifact["name"]).stat().st_size == artifact["size_bytes"] and
                       file_sha256(root / artifact["name"]) == "sha256:" + artifact["sha256"]
                       for artifact in artifacts)
        integrity = "verified" if verified else "corrupt" if present else "absent"
        model_packs[pack_id] = {"integrity": integrity,
                                "reason_codes": [f"model.{pack_id}.{integrity}"]}
    result = {
        "schema_version": 1, "facts_version": FACTS_VERSION, "captured_at": utc_now(),
        "os": os_facts(),
        "installation": {"classification": classification, "reason_codes": install_reasons,
            "manifest": {"exists": manifest is not None, "health": classification,
                         "schema_version": manifest.get("schema_version") if manifest else None},
            "active_version": active, "previous_version": previous, "target_version": target_version,
            "version_relation": version_relation(active, target_version),
            "desired_configuration": desired, "payloads": []},
        "paths": {"data_root": writable_path_fact(r["data"]),
            "cache_root": writable_path_fact(r["cache"]),
            "config_root": writable_path_fact(r["config"]),
            "free_bytes": disk, "required_bytes": {"download": 0, "build": 4_000_000_000, "staging": 1_000_000_000}},
        "toolchain": {"tools": {name: command_exists(name) for name in ("cmake", "ninja", "c++", "pkg-config")},
            "build_dependencies": ["cmake", "ninja", "c++", "pkg-config"],
            "missing_build_dependencies": [name for name in ("cmake", "ninja", "c++", "pkg-config") if not command_exists(name)],
            "reason_codes": []},
        "runtime": {"dependencies": {}, "onnxruntime": ort},
        "systemd_user": {"systemctl_present": systemd_present, "manager_usable": systemd_usable,
            "reason_codes": [systemd_reason]},
        "v4l2": {"package_state": package_state, "module_available": module_available,
            "module_loaded": module_loaded, "device_present": device_present, "device_number": 10,
            "device_label": device_label, "device_number_conflict": device_conflict, "persistence_state": persistence_state,
            "kernel_headers_usable": headers, "dkms_viable": module_available or (headers and dkms_present),
            "secure_boot": secure_boot, "owned_configuration": owned_v4l,
            "owned_configuration_validated": len(owned_v4l) == len(V4L_OWNED_PATHS),
            "reason_codes": ["v4l.module.available" if module_available else "v4l.module.unavailable",
                             "v4l.module.loaded" if module_loaded else "v4l.module.not_loaded",
                             "v4l.device.present" if device_present else "v4l.device.absent",
                             "v4l.dkms.viable" if module_available or (headers and dkms_present) else "v4l.dkms.not_viable",
                             f"secure_boot.{secure_boot}"]},
        "privileged_helper": helper_facts(),
        "gpus": {"devices": gpu_devices, "nvidia": nvidia, "vulkan": vulkan},
        "maxine": maxine,
        "effects": {"capabilities": effects},
        "models": {"packs": model_packs, "default_pack_ids": list(DEFAULT_MODEL_PACKS),
                   "catalog_size_summary": catalog_size_summary(catalog, catalog_hash)},
        "cache": {"release_artifacts": {}, "model_artifacts": {
            pack["id"]: {"verified": all(
                (r["cache"] / "models/sha256" / artifact["sha256"]).is_file() and
                not (r["cache"] / "models/sha256" / artifact["sha256"]).is_symlink() and
                file_sha256(r["cache"] / "models/sha256" / artifact["sha256"]) ==
                "sha256:" + artifact["sha256"] for artifact in pack["artifacts"])}
            for pack in catalog["packs"]}},
        "connectivity": {"release_source": {"state": "offline" if os.environ.get("STUDIOCAST_INSTALLER_OFFLINE") == "1" else "unknown",
                "reason_codes": ["network.release.offline_requested" if os.environ.get("STUDIOCAST_INSTALLER_OFFLINE") == "1" else "network.release.not_probed"]},
            "model_source": {"state": "offline" if os.environ.get("STUDIOCAST_INSTALLER_OFFLINE") == "1" else "unknown",
                "reason_codes": ["network.models.offline_requested" if os.environ.get("STUDIOCAST_INSTALLER_OFFLINE") == "1" else "network.models.not_probed"]}},
        "reason_codes": [],
    }
    stable = copy.deepcopy(result)
    stable.pop("captured_at", None)
    result["host_fingerprint"] = digest(stable)
    return result


def normalize_fixture(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != 1 or value.get("facts_version", FACTS_VERSION) != FACTS_VERSION:
        raise BackendError("facts.unsupported_schema", "Fixture is not installer facts v1")
    out = copy.deepcopy(value)
    out.setdefault("facts_version", FACTS_VERSION)
    out.setdefault("captured_at", "fixture")
    stable = copy.deepcopy(out)
    stable.pop("host_fingerprint", None)
    stable.pop("captured_at", None)
    out["host_fingerprint"] = digest(stable)
    return out


def integrate_release_facts(facts: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    evidence = getattr(args, "verified_release", None)
    if evidence is None:
        return facts
    out = copy.deepcopy(facts)
    offline = bool(getattr(args, "offline", False) or evidence.get("offline") is True)
    archive_state = evidence.get("archive_state")
    verified_local = (archive_state == "candidate_verified" or
                      evidence.get("archive_cache_state") == "cached_verified")
    out.setdefault("cache", {}).setdefault("release_artifacts", {})["studiocast-source"] = {
        "verified": verified_local,
        "state": evidence.get("archive_cache_state") if verified_local else archive_state,
        "version": evidence.get("version"), "sha256": evidence.get("archive_sha256"),
        "path": (evidence.get("archive_candidate_path")
                 if archive_state == "candidate_verified" else evidence.get("archive_path")),
        "reason_codes": ["cache.release.verified" if verified_local else
                         "cache.release.acquisition_required"],
    }
    if offline:
        out.setdefault("connectivity", {})["release_source"] = {
            "state": "offline", "reason_codes": ["network.release.offline_requested"]}
        out["connectivity"]["model_source"] = {
            "state": "offline", "reason_codes": ["network.models.offline_requested"]}
    # Release evidence is independently signature/hash bound in the plan and
    # revalidated by release_acquire.  Keep the machine-facts fingerprint
    # stable so Apply can consume the exact fixture/host facts reviewed by the
    # analyzer without requiring a receipt as an ambient bearer token.
    return out
