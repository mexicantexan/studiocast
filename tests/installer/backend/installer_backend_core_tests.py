#!/usr/bin/env python3
"""Hermetic contract tests for the StudioCast installer backend."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import py_compile
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SOURCE = Path(os.environ.get("STUDIOCAST_SOURCE_DIR", Path(__file__).resolve().parents[3]))
BACKEND = SOURCE / "installer/backend/studiocast-installer-backend"
BACKEND_PACKAGE = SOURCE / "installer/backend/studiocast_installer_backend"
sys.path.insert(0, str(BACKEND_PACKAGE.parent))
from studiocast_installer_backend.typed_video_status import (  # noqa: E402
    parse_typed_multi_island_video_status,
)

BACKEND_MODULES = (
    "__init__.py", "cli.py", "contracts.py", "executor.py", "facts.py",
    "manifest.py", "packaged.py", "plan.py", "recommendation.py",
    "typed_video_status.py",
)
DEFAULT_PACKS = [
    "fastenhancer_s_vd_v1", "fastenhancer_m_vd_v1", "modnet-webnn-256-fp32",
    "yunet_opencv_zoo_2023mar_fp32", "dlib_68_ibug_300w",
    "gaze_correction_cam_flx_v0_1_1", "fastdvdnet_sigma15",
]
MODEL_CATALOG_SHA256 = "sha256:d2b23cab9ddad91adfe5cac4b2ad4433fa624714d40cb0d7ebdb8deebe93414d"
MODEL_PACK_SIZE_ROWS = [
    ("fastenhancer_s_vd_v1", 1, 832775),
    ("fastenhancer_m_vd_v1", 1, 2033628),
    ("modnet-webnn-256-fp32", 1, 25888640),
    ("yunet_opencv_zoo_2023mar_fp32", 1, 232589),
    ("dlib_68_ibug_300w", 1, 99693937),
    ("gaze_correction_cam_flx_v0_1_1", 2, 2125930),
    ("fastdvdnet_sigma15", 1, 416788),
]
CANONICAL_EFFECTS = [
    "mirror", "virtual_background.blur", "virtual_background.remove",
    "virtual_background.replace", "auto_frame", "eye_contact",
    "video_noise_removal", "virtual_key_light", "vignette",
]


def stage_backend(installer_root: Path) -> Path:
    staged_backend = installer_root / "studiocast-installer-backend"
    installer_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BACKEND, staged_backend)
    staged_backend.chmod(0o755)
    shutil.copytree(
        BACKEND_PACKAGE,
        installer_root / BACKEND_PACKAGE.name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return staged_backend


def terminal_result(text: str) -> dict:
    marker = text.rfind("\n{")
    value = json.loads(text[marker + 1 if marker >= 0 else 0:])
    if value.get("result_version") != "installer-result/v1":
        raise AssertionError("terminal installer result is missing")
    return value


def catalog_size_summary() -> dict:
    return {
        "schema_version": 1,
        "catalog_version": "2026-07-19",
        "catalog_sha256": MODEL_CATALOG_SHA256,
        "trusted": True,
        "reason_code": "artifact_sizes_independently_verified_against_pinned_sha256",
        "pack_count": len(MODEL_PACK_SIZE_ROWS),
        "artifact_count": sum(row[1] for row in MODEL_PACK_SIZE_ROWS),
        "download_bytes": sum(row[2] for row in MODEL_PACK_SIZE_ROWS),
        "packs": [
            {"id": pack_id, "artifact_count": artifact_count,
             "download_bytes": download_bytes, "size_status": "known"}
            for pack_id, artifact_count, download_bytes in MODEL_PACK_SIZE_ROWS
        ],
    }


def progress_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event_version") == "installer-progress/v1":
            events.append(value)
    return events


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
        "models": {"packs": {}, "default_pack_ids": list(DEFAULT_PACKS),
                   "catalog_size_summary": catalog_size_summary()},
        "cache": {"release_artifacts": {}, "model_artifacts": {}},
        "connectivity": {"release_source": {"state": "online", "reason_codes": []},
                         "model_source": {"state": "online", "reason_codes": []}},
        "reason_codes": [],
    }


def canonical_capabilities(vulkan_profile="unavailable", blocker="vulkan_no_physical_device"):
    capabilities = {}
    for effect in (
        "mirror", "virtual_background.blur", "virtual_background.remove",
        "virtual_background.replace", "auto_frame", "eye_contact",
        "video_noise_removal", "virtual_key_light", "vignette",
    ):
        ready = vulkan_profile == "valid" and effect in {"mirror", "vignette"}
        parameter_contract = "fixed_center" if effect == "vignette" else None
        specific_blocker = blocker
        if vulkan_profile == "valid":
            specific_blocker = {
                "auto_frame": "vulkan_auto_frame_yunet_unavailable",
                "virtual_background.blur": "open_vulkan_matting_unavailable",
                "virtual_background.remove": "open_vulkan_matting_unavailable",
                "virtual_background.replace": "open_vulkan_matting_unavailable",
                "virtual_key_light": "open_vulkan_matting_unavailable",
                "eye_contact": "open_vulkan_eye_contact_unavailable",
                "video_noise_removal": "open_vulkan_video_noise_removal_unavailable",
            }.get(effect, blocker)
        capabilities[effect] = {
            "maxine": "unavailable", "cuda": "unavailable",
            "vulkan": ("production_usable" if ready else
                       "experimental" if effect in {
                           "virtual_background.blur", "virtual_background.remove",
                           "virtual_background.replace", "virtual_key_light"}
                       else "unavailable"),
            "cpu": "unavailable",
            "success_reason_codes": {
                "maxine": [], "cuda": [],
                "vulkan": (["open_vulkan_mirror_production_ready"] if ready and effect == "mirror"
                            else ["open_vulkan_vignette_fixed_center_production_ready"]
                            if ready else []),
                "cpu": [],
            },
            "blocker_reason_codes": {
                "maxine": ["maxine.effect.unavailable"],
                "cuda": ["open_cuda_effect_unavailable"],
                "vulkan": [] if ready else [specific_blocker],
                "cpu": [f"effect.{effect}.cpu.no_selectable_production_path"],
            },
            "cpu_evidence": {
                "selectable_production_path": False,
                "reason_code": f"effect.{effect}.cpu.no_selectable_production_path",
            },
            "vulkan_evidence": {
                "runtime_diagnostics_available": vulkan_profile != "unavailable",
                "backend_compiled": vulkan_profile != "unavailable",
                "runtime_library_found": vulkan_profile not in {"unavailable", "no_gpu"},
                "instance_created": vulkan_profile == "valid",
                "physical_device_found": vulkan_profile == "valid",
                "non_cpu_device_selected": vulkan_profile == "valid",
                "compute_queue_available": vulkan_profile == "valid",
                "logical_device_created": vulkan_profile == "valid",
                "context_created": vulkan_profile == "valid",
                "context_healthy": vulkan_profile == "valid",
                "production_hardware_ready": vulkan_profile == "valid",
                "utility_kernels_ready": vulkan_profile == "valid",
                "common_ready": vulkan_profile == "valid",
                "live_stage_implemented": effect not in {
                    "eye_contact", "video_noise_removal"},
                "daemon_production_ready": ready,
                "effect_production_ready_attested": ready,
                "production_ready": ready,
                "runtime_ready": ready,
                "model_artifact_ready": ready,
                "warmup_complete": ready,
                "exact_device_and_queue": ready,
                "device_resident": ready,
                "synchronization_contract_validated": ready,
                "bounded_reusable_allocations": ready,
                "parity_validated": ready,
                "no_cpu_layers": ready,
                "no_dynamic_allocations": ready,
                "no_cpu_readback": ready,
                "no_cpu_fallback": ready,
                "temporal_history_device_resident": False,
                "temporal_history_bounded": False,
                "history_reset_on_disable": False,
                "history_reset_on_reconfigure": False,
                "capture_sequence_discontinuity_reset": False,
                "parameter_contract": parameter_contract,
                "selectable_cpu_fallback": False,
            },
        }
    return capabilities


def promote_production_looking_vulkan(facts, effect):
    matrix = facts["effects"]["capabilities"][effect]
    matrix["vulkan"] = "production_usable"
    matrix["blocker_reason_codes"]["vulkan"] = []
    matrix["success_reason_codes"]["vulkan"] = [{
        "virtual_background.blur": "open_vulkan_matting_production_ready",
        "eye_contact": "open_vulkan_eye_contact_production_ready",
        "video_noise_removal": "open_vulkan_video_noise_removal_production_ready",
    }[effect]]
    evidence = matrix["vulkan_evidence"]
    for key in (
        "live_stage_implemented", "daemon_production_ready",
        "effect_production_ready_attested", "production_ready",
        "runtime_ready", "model_artifact_ready", "warmup_complete",
        "exact_device_and_queue", "device_resident",
        "synchronization_contract_validated", "bounded_reusable_allocations",
        "parity_validated", "no_cpu_readback", "no_cpu_fallback",
    ):
        evidence[key] = True
    if effect == "virtual_background.blur":
        evidence["no_cpu_layers"] = True
        evidence["no_dynamic_allocations"] = True
    if effect == "video_noise_removal":
        for key in (
            "temporal_history_device_resident", "temporal_history_bounded",
            "history_reset_on_disable", "history_reset_on_reconfigure",
            "capture_sequence_discontinuity_reset",
        ):
            evidence[key] = True
    return matrix


def typed_pair(identifier, token):
    return {"id": identifier, "token": token}


def typed_domain(backend_id, adapter_id):
    values = {
        2: ("cuda", 2, "cuda_driver", 2, "cuda_driver", 2,
            "cuda_device_pointer", 2, "cuda_stream_ordered"),
        3: ("maxine", 3, "maxine_nvcv", 3, "maxine_nvcv", 3,
            "nvcv_image", 3, "maxine_bound_cuda_stream"),
        5: ("vulkan", 5, "vulkan", 5, "vulkan", 5,
            "vulkan_buffer", 5, "vulkan_queue_fence"),
    }
    (backend, api_id, api, allocator_id, allocator, memory_id, memory,
     sync_id, sync) = values[backend_id]
    return {
        "backend": typed_pair(backend_id, backend),
        "api_family": typed_pair(api_id, api),
        "physical_device": {
            "vendor_id": 0x10DE, "device_id": 0x1C30,
            "stable_id": f"{adapter_id:02x}" + "00" * 15,
        },
        "logical_context_generation": 14,
        "allocator_family": typed_pair(allocator_id, allocator),
        "memory_handle_family": typed_pair(memory_id, memory),
        "queue_or_stream_identity": 1000 + adapter_id,
        "ownership_model": typed_pair(1, "adapter_exclusive"),
        "synchronization_protocol": typed_pair(sync_id, sync),
        "adapter_instance_id": adapter_id,
    }


def typed_invalidation():
    return {
        "generations": {
            "configuration": 11, "geometry": 21, "model": 22,
            "provider": 12, "device": 13, "context": 14,
        },
        "canonical_plan_hash": 31,
        "model_artifact_hash": 32,
        "execution_domain_hash": 33,
    }


def typed_capability(effect_id, effect, lane_id=0, lane="", code_id=0,
                     code="", provider_id=0, provider="", status_id=0,
                     status="invalid", *, evidence=None):
    semantic_effect = typed_pair(effect_id, effect) if lane_id else typed_pair(0, "")
    live = lane_id != 0 and status_id in {1, 3}
    return {
        "lane": typed_pair(lane_id, lane),
        "code": typed_pair(code_id, code),
        "nested_code": typed_pair(0, ""),
        "semantic": {
            "effect": semantic_effect,
            "provider": typed_pair(provider_id, provider),
            "status": typed_pair(status_id, status),
            "reason": typed_pair(0, ""),
            "cpu_tail": typed_pair(0, ""),
            "cpu_tail_scratch_bytes": 0,
            "cpu_tail_placement": typed_pair(1, "after_stage"),
            "live_stage": live,
            "production_eligible": live,
            "exact_evidence": live,
            "supports_tracked_vignette_center": False,
            "evidence": list(evidence or []),
        },
    }


def typed_vulkan_evidence(effect_id, effect):
    tokens = list(enumerate([
        "vulkan_build_enabled", "vulkan_loader_available",
        "vulkan_physical_device_available", "vulkan_non_cpu_device",
        "vulkan_compute_queue_available", "vulkan_context_healthy",
        "vulkan_utility_kernels_ready", "vulkan_exact_device_shared",
        "vulkan_device_local_residency", "vulkan_bounded_allocations",
        "vulkan_no_cpu_layers", "vulkan_no_intermediate_readbacks",
    ], start=1))
    tokens.append(
        (13, "open_vulkan_mirror_production_ready") if effect == "mirror"
        else (14, "open_vulkan_vignette_fixed_center_production_ready"))
    return [{
        "code": typed_pair(index, token),
        "status": typed_pair(1, "ready"), "reason": typed_pair(0, ""),
        "effect": typed_pair(effect_id, effect),
        "backend": typed_pair(5, "vulkan"), "satisfied": True,
    } for index, token in tokens]


def typed_matrix_lane(effect_id, effect, lane_id, lane, capability,
                      selectable=False):
    semantic = capability["semantic"]
    return {
        "effect": effect, "effect_id": effect_id, "lane": lane,
        "lane_id": lane_id,
        "disposition": "production_usable" if selectable else "unavailable",
        "code": capability["code"],
        "nested_code": capability["nested_code"],
        "selectable": selectable,
        "semantic": {
            "provider": semantic["provider"]["token"],
            "provider_id": semantic["provider"]["id"],
            "status": semantic["status"]["token"],
            "status_id": semantic["status"]["id"],
            "reason": semantic["reason"]["token"],
            "reason_id": semantic["reason"]["id"],
            "live_stage": semantic["live_stage"],
            "production_eligible": semantic["production_eligible"],
            "exact_evidence": semantic["exact_evidence"],
            "cpu_tail": semantic["cpu_tail"]["token"],
            "cpu_tail_scratch_bytes": semantic["cpu_tail_scratch_bytes"],
            "supports_tracked_vignette_center":
                semantic["supports_tracked_vignette_center"],
            "evidence": [{
                "code_id": item["code"]["id"],
                "code": item["code"]["token"],
                "status_id": item["status"]["id"],
                "status": item["status"]["token"],
                "reason_id": item["reason"]["id"],
                "reason": item["reason"]["token"],
                "effect": item["effect"]["token"],
                "backend": item["backend"]["token"],
                "satisfied": item["satisfied"],
            } for item in semantic["evidence"]],
        },
    }


def typed_capability_lanes(effects):
    unavailable = {
        1: (2, "maxine.effect.unavailable"),
        2: (4, "open_cuda_effect_unavailable"),
        3: (116, "open_vulkan_runtime_diagnostics_unavailable"),
    }
    cpu_codes = {
        1: "effect.video_noise_removal.cpu.no_selectable_production_path",
        2: "effect.eye_contact.cpu.no_selectable_production_path",
        3: "effect.virtual_background.blur.cpu.no_selectable_production_path",
        4: "effect.virtual_background.remove.cpu.no_selectable_production_path",
        5: "effect.virtual_background.replace.cpu.no_selectable_production_path",
        6: "effect.virtual_key_light.cpu.no_selectable_production_path",
        7: "effect.auto_frame.cpu.no_selectable_production_path",
        8: "effect.vignette.cpu.no_selectable_production_path",
        9: "effect.mirror.cpu.no_selectable_production_path",
    }
    lanes = []
    for value in effects:
        effect_id = value["effect"]["id"]
        effect = value["effect"]["token"]
        for lane_id, lane in (
            (1, "maxine"), (2, "open_cuda"), (3, "open_vulkan"), (4, "cpu"),
        ):
            selected = value["selected"] and (
                value["selected_capability"]["lane"]["id"] == lane_id)
            if selected:
                capability = value["selected_capability"]
            elif lane_id == 4:
                capability = typed_capability(
                    effect_id, effect, lane_id, lane, 9 + effect_id,
                    cpu_codes[effect_id], 0, "", 4, "unavailable")
                capability["semantic"]["live_stage"] = False
                capability["semantic"]["production_eligible"] = False
                capability["semantic"]["exact_evidence"] = False
                capability["semantic"]["reason"] = typed_pair(
                    27, "semantic_effect_unsupported")
            else:
                code_id, token = unavailable[lane_id]
                capability = typed_capability(
                    effect_id, effect, lane_id, lane, code_id, token, 0, "",
                    4, "unavailable")
                capability["semantic"]["live_stage"] = False
                capability["semantic"]["production_eligible"] = False
                capability["semantic"]["exact_evidence"] = False
                capability["semantic"]["reason"] = typed_pair(
                    27, "semantic_effect_unsupported")
            lanes.append(typed_matrix_lane(
                effect_id, effect, lane_id, lane, capability, selected))
    return lanes


def typed_island(route_id, route, island_id, adapter_id, stage_id,
                 effect_id, effect, backend_id, backend, provider_id, provider,
                 cpu_tails=0):
    domain = typed_domain(backend_id, adapter_id)
    transfers = {
        "uploads": 1, "final_readbacks": 1, "intermediate_readbacks": 0,
        "intermediate_uploads": 0, "synchronizations": 1,
        "cpu_tail_boundaries": cpu_tails,
    }
    stage = {
        "stage_id": stage_id, "effect": typed_pair(effect_id, effect),
        "backend": typed_pair(backend_id, backend),
        "provider": typed_pair(provider_id, provider),
        "status": typed_pair(1, "ready"),
        "reason": typed_pair(0, ""),
        "failure_action": typed_pair(2, "fail_open_preserve_input"),
        "committed_status": typed_pair(1, "succeeded"),
        "required": False, "resolved": True, "committed": True,
    }
    resolution_stage = {
        "stage_id": stage_id, "effect": typed_pair(effect_id, effect),
        "required": False,
        "failure_action": typed_pair(2, "fail_open_preserve_input"),
        "requires_tracked_vignette_center": False,
        "status": typed_pair(1, "ready"), "reason": typed_pair(0, ""),
        "selected_backend": typed_pair(backend_id, backend),
        "provider": typed_pair(provider_id, provider),
        "adapter_instance_id": adapter_id, "execution_domain": domain,
        "cpu_tail": typed_pair(0, ""), "cpu_tail_scratch_bytes": 0,
        "cpu_tail_placement": typed_pair(1, "after_stage"),
        "skipped_optional": False, "fallback_selected": False,
        "evidence": [],
    }
    return {
        "route": typed_pair(route_id, route),
        "island_id": island_id, "adapter_instance_id": adapter_id,
        "snapshot_generation": 100, "capability_generation": 10,
        "backend": typed_pair(backend_id, backend),
        "execution_domain": domain,
        "resolution": {
            "request_mode": typed_pair(2, "explicit_backend"),
            "strict_backend_selection": True,
            "status": typed_pair(1, "ready"), "reason": typed_pair(0, ""),
            "requested_backend": typed_pair(backend_id, backend),
            "selected_backend": typed_pair(backend_id, backend),
            "active_backend": typed_pair(backend_id, backend),
            "selected_provider": typed_pair(provider_id, provider),
            "available": True, "degraded": False, "fallback": False,
            "passthrough": False, "stages": [resolution_stage],
        },
        "readiness": {
            "configured": True, "program_id": 100 + island_id,
            "invalidation_key": typed_invalidation(),
            "status": typed_pair(1, "ready"), "reason": typed_pair(0, ""),
            "requested_backend": typed_pair(backend_id, backend),
            "selected_backend": typed_pair(backend_id, backend),
            "provider": typed_pair(provider_id, provider),
            "execution_domain": domain,
            "ready_stage_mask": 1, "blocked_stage_mask": 0,
            "disabled_stage_mask": 0, "available": True,
            "degraded": False, "rebuild_requested": False, "evidence": [],
        },
        "telemetry": {
            "requested_backend": typed_pair(backend_id, backend),
            "selected_backend": typed_pair(backend_id, backend),
            "active_backend": typed_pair(backend_id, backend),
            "active_provider": typed_pair(provider_id, provider),
            "runtime_status": typed_pair(6, "executed"),
            "runtime_reason": typed_pair(0, ""),
            "transfers": transfers,
        },
        "latest_committed_frame": {
            "valid": True, "program_id": 100 + island_id,
            "invalidation_key": typed_invalidation(),
            "execution_domain": domain,
            "capture_sequence": 501, "output_commit_sequence": 9,
            "stages": [{
                "stage_id": stage_id, "effect": typed_pair(effect_id, effect),
                "backend": typed_pair(backend_id, backend),
                "provider": typed_pair(provider_id, provider),
                "status": typed_pair(1, "succeeded"),
                "reason": typed_pair(0, ""),
            }],
        },
        "stages": [stage], "valid": True, "current": True,
    }


def typed_effect(effect_id, effect):
    invalid = typed_capability(effect_id, effect)
    return {
        "effect": typed_pair(effect_id, effect),
        "requested_capability": invalid,
        "selected_capability": invalid,
        "requested_backend": typed_pair(0, ""),
        "requested_provider": typed_pair(0, ""),
        "selected_backend": typed_pair(0, ""),
        "selected_provider": typed_pair(0, ""),
        "active_backend": typed_pair(0, ""),
        "active_provider": typed_pair(0, ""),
        "status": typed_pair(5, "passthrough"),
        "reason": typed_pair(0, ""),
        "failure_action": typed_pair(2, "fail_open_preserve_input"),
        "commit": {
            "valid": False, "capture_sequence": 0,
            "output_commit_sequence": 0,
        },
        "selected_island_id": 0, "committed_stage_id": 0,
        "committed_stage_status": typed_pair(0, "not_run"),
        "strict_backend_selection": False,
        "strict_provider_selection": False, "required": False,
        "requested": False, "available": False, "selected": False,
        "active": False, "degraded": False, "blocked": False,
        "fallback": False,
    }


def active_typed_effect(value, lane_id, lane, code_id, code, island_id,
                        stage_id, backend_id, backend, provider_id, provider):
    effect_id = value["effect"]["id"]
    effect = value["effect"]["token"]
    evidence = (
        typed_vulkan_evidence(effect_id, effect)
        if lane_id == 3 else [])
    capability = typed_capability(
        effect_id, effect, lane_id, lane, code_id, code,
        provider_id, provider, 1, "ready", evidence=evidence)
    value.update({
        "requested_capability": capability,
        "selected_capability": capability,
        "requested_backend": typed_pair(backend_id, backend),
        "requested_provider": typed_pair(provider_id, provider),
        "selected_backend": typed_pair(backend_id, backend),
        "selected_provider": typed_pair(provider_id, provider),
        "active_backend": typed_pair(backend_id, backend),
        "active_provider": typed_pair(provider_id, provider),
        "status": typed_pair(6, "executed"),
        "commit": {
            "valid": True, "capture_sequence": 501,
            "output_commit_sequence": 9,
        },
        "selected_island_id": island_id, "committed_stage_id": stage_id,
        "committed_stage_status": typed_pair(1, "succeeded"),
        "strict_backend_selection": True,
        "strict_provider_selection": True,
        "requested": True, "available": True, "selected": True, "active": True,
    })


def typed_multi_island_fixture():
    effect_order = [
        (1, "video_noise_removal"), (2, "eye_contact"),
        (3, "virtual_background.blur"), (4, "virtual_background.remove"),
        (5, "virtual_background.replace"), (6, "virtual_key_light"),
        (7, "auto_frame"), (8, "vignette"), (9, "mirror"),
    ]
    effects = [typed_effect(identifier, token) for identifier, token in effect_order]
    by_name = {effect["effect"]["token"]: effect for effect in effects}
    active_typed_effect(
        by_name["mirror"], 3, "open_vulkan", 100,
        "open_vulkan_mirror_production_ready", 1, 11, 5, "vulkan", 5,
        "open_vulkan_production_kernels")
    active_typed_effect(
        by_name["eye_contact"], 1, "maxine", 1, "maxine.effect.usable",
        2, 12, 3, "maxine", 2, "maxine_ar")
    active_typed_effect(
        by_name["virtual_background.blur"], 2, "open_cuda", 3,
        "open_cuda_effect_available", 3, 13, 2, "cuda", 3,
        "open_video_onnxruntime_cuda")
    islands = [
        typed_island(1, "final_visual", 1, 101, 11, 9, "mirror", 5,
                     "vulkan", 5, "open_vulkan_production_kernels"),
        typed_island(2, "maxine_resident", 2, 102, 12, 2, "eye_contact", 3,
                     "maxine", 2, "maxine_ar", 1),
        typed_island(
            3, "open_cuda_virtual_background", 3, 103, 13, 3,
            "virtual_background.blur", 2, "cuda", 3,
            "open_video_onnxruntime_cuda", 1),
    ]
    return {
        "valid": True, "schema_version": 1, "hal_contract_version": 1,
        "reason_code_version": 1, "status_generation": 100,
        "generations": {
            "capability": 10, "configuration": 11, "provider": 12,
            "device": 13, "context": 14,
        },
        "program_generations": typed_invalidation()["generations"],
        "authority": {
            "scope": "prepared_routes_only", "complete": True,
            "prepared_route_effect_mask": 262,
            "compatibility_uncovered_effect_mask": 0,
        },
        "lifecycle": typed_pair(3, "running"),
        "capabilities": {
            "valid": True, "schema_version": 1, "hal_contract_version": 1,
            "reason_code_version": 1,
            "generations": {
                "capability": 10, "configuration": 11, "provider": 12,
                "device": 13, "context": 14,
            },
            "refresh": {
                "status": "succeeded", "current": True,
                "code": typed_pair(200, "capability_snapshot_current"),
                "requested_generation": 10, "completed_generation": 10,
            },
            "lanes": typed_capability_lanes(effects),
        },
        "writer_commit": {
            "valid": True, "capture_sequence": 501,
            "output_commit_sequence": 9,
        },
        "islands": islands, "effects": effects,
        "aggregate": {
            "current_island_count": 3,
            "transfers": {
                "uploads": 3, "final_readbacks": 3,
                "intermediate_readbacks": 0, "intermediate_uploads": 0,
                "synchronizations": 3, "cpu_tail_boundaries": 2,
            },
        },
    }


def machine_recommendation_facts(vendor, vulkan_profile="valid",
                                 blocker="vulkan_no_physical_device"):
    facts = base_facts()
    facts["effects"]["capabilities"] = canonical_capabilities(
        vulkan_profile, blocker)
    if vendor == "hybrid":
        facts["gpus"]["devices"] = [
            {"id": "integrated", "vendor": "intel"},
            {"id": "discrete", "vendor": "nvidia"},
        ]
    elif vendor != "none":
        facts["gpus"]["devices"] = [{"id": "gpu0", "vendor": vendor}]
    facts["gpus"]["vulkan"] = {
        "loader_present": vulkan_profile != "no_gpu",
        "physical_devices": facts["gpus"]["devices"],
        "compute_device_usable": vulkan_profile == "valid",
        "reason_codes": [blocker] if vulkan_profile != "valid" else
                        ["vulkan_compute_device_usable"],
    }
    return facts


class Sandbox:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="studiocast-installer-core-")
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env.pop("PYTHONPATH", None)
        self.env.update({"HOME": str(self.root / "home"), "XDG_DATA_HOME": str(self.root / "data"),
                         "XDG_CONFIG_HOME": str(self.root / "config"), "XDG_STATE_HOME": str(self.root / "state"),
                         "XDG_CACHE_HOME": str(self.root / "cache"),
                         "STUDIOCAST_INSTALLER_OFFLINE": "1",
                         "STUDIOCAST_INSTALLER_TEST_MODE": "1",
                         "PYTHONDONTWRITEBYTECODE": "1"})
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

    def test_backend_package_has_exact_compilable_importable_topology(self):
        self.assertEqual(set(BACKEND_MODULES),
                         {path.name for path in BACKEND_PACKAGE.iterdir()
                          if path.is_file() and path.suffix == ".py"})
        compile_root = self.s.root / "compiled"
        compile_root.mkdir()
        for name in BACKEND_MODULES:
            py_compile.compile(
                str(BACKEND_PACKAGE / name),
                cfile=str(compile_root / f"{name}c"),
                doraise=True,
            )
        imports = ";".join(
            f"import studiocast_installer_backend.{Path(name).stem}"
            for name in BACKEND_MODULES if name != "__init__.py"
        )
        code = "import sys;sys.path.insert(0,sys.argv[1]);" + imports
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(BACKEND.parent)],
            cwd=self.s.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_typed_multi_island_status_accepts_exact_current_evidence_shadow_only(self):
        parsed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": typed_multi_island_fixture(),
        })
        self.assertTrue(parsed["present"] and parsed["valid"])
        self.assertEqual(parsed["mode"], "shadow_only")
        self.assertFalse(parsed["recommendation_eligible"])
        self.assertEqual(parsed["authority"], {
            "scope": "prepared_routes_only", "complete": True,
            "prepared_route_effect_mask": 262,
            "compatibility_uncovered_effect_mask": 0,
        })
        self.assertEqual(set(parsed["effects"]), set(CANONICAL_EFFECTS))
        self.assertEqual(
            [(entry["route"], entry["backend"], entry["provider"])
             for entry in parsed["islands"]],
            [
                ("final_visual", "vulkan",
                 "open_vulkan_production_kernels"),
                ("maxine_resident", "maxine", "maxine_ar"),
                ("open_cuda_virtual_background", "cuda",
                 "open_video_onnxruntime_cuda"),
            ],
        )
        self.assertEqual(parsed["aggregate"]["transfers"], {
            "uploads": 3, "final_readbacks": 3,
            "intermediate_readbacks": 0, "intermediate_uploads": 0,
            "synchronizations": 3, "cpu_tail_boundaries": 2,
        })
        self.assertEqual(
            parsed["effects"]["mirror"]["requested_capability"]["code"],
            typed_pair(100, "open_vulkan_mirror_production_ready"))
        self.assertEqual(
            parsed["effects"]["virtual_background.blur"]["active_provider"],
            "open_video_onnxruntime_cuda")

    def test_typed_multi_island_status_rejects_versions_effect_topology_and_staleness(self):
        cases = []
        for field, reason in (
            ("schema_version", "typed_status.unsupported_schema_version"),
            ("hal_contract_version",
             "typed_status.unsupported_hal_contract_version"),
            ("reason_code_version",
             "typed_status.unsupported_reason_code_version"),
        ):
            value = typed_multi_island_fixture()
            value[field] = 99
            cases.append((value, reason))
        missing = typed_multi_island_fixture()
        missing["effects"].pop()
        cases.append((missing, "typed_status.effect_count_invalid"))
        duplicate = typed_multi_island_fixture()
        duplicate["effects"][-1] = copy.deepcopy(duplicate["effects"][0])
        cases.append((duplicate, "typed_status.duplicate_effect"))
        unknown = typed_multi_island_fixture()
        unknown["effects"][-1]["effect"] = typed_pair(99, "future_effect")
        cases.append((unknown, "typed_status.effect_invalid"))
        stale = typed_multi_island_fixture()
        stale["islands"][0]["snapshot_generation"] -= 1
        cases.append((stale, "typed_status.stale_generation"))
        mixed_generation = typed_multi_island_fixture()
        mixed_generation["program_generations"]["provider"] += 1
        cases.append((mixed_generation, "typed_status.mixed_generation"))
        mixed_commit = typed_multi_island_fixture()
        mixed_commit["islands"][1]["latest_committed_frame"][
            "output_commit_sequence"] += 1
        cases.append((mixed_commit, "typed_status.mixed_writer_commit"))
        invalid_commit = typed_multi_island_fixture()
        invalid_commit["writer_commit"]["output_commit_sequence"] = 0
        cases.append((invalid_commit, "typed_status.invalid_commit"))
        aggregate = typed_multi_island_fixture()
        aggregate["aggregate"]["transfers"]["uploads"] += 1
        cases.append((aggregate, "typed_status.aggregate_mismatch"))
        missing_authority = typed_multi_island_fixture()
        missing_authority.pop("authority")
        cases.append((
            missing_authority, "typed_status.authority_invalid"))
        malformed_authority = typed_multi_island_fixture()
        malformed_authority["authority"]["prepared_route_effect_mask"] = 0
        cases.append((
            malformed_authority, "typed_status.authority_invalid"))
        incomplete_authority = typed_multi_island_fixture()
        incomplete_authority["authority"].update({
            "complete": False,
            "compatibility_uncovered_effect_mask": 1 << 0,
        })
        cases.append((
            incomplete_authority, "typed_status.authority_incomplete"))
        domain = typed_multi_island_fixture()
        domain["islands"][2]["execution_domain"]["api_family"] = typed_pair(
            5, "vulkan")
        cases.append((domain, "typed_status.island_invalid"))
        strict = typed_multi_island_fixture()
        strict_effect = next(
            effect for effect in strict["effects"]
            if effect["effect"]["token"] == "virtual_background.blur")
        strict_effect["requested_backend"] = typed_pair(
            4, "open_video_onnxruntime")
        cases.append((strict, "typed_status.strict_backend_substitution"))

        for value, reason in cases:
            with self.subTest(reason=reason):
                parsed = parse_typed_multi_island_video_status({
                    "typed_multi_island_video_status": value,
                })
                self.assertTrue(parsed["present"])
                self.assertFalse(parsed["valid"])
                self.assertFalse(parsed["recommendation_eligible"])
                self.assertEqual(parsed["reason_codes"], [reason])
                self.assertEqual(parsed["effects"], {})

    def test_typed_multi_island_status_rejects_uncovered_compatibility_blockers(self):
        value = typed_multi_island_fixture()
        effects = {
            entry["effect"]["token"]: entry for entry in value["effects"]
        }
        vignette = effects["vignette"]
        vignette_capability = typed_capability(
            8, "vignette", 3, "open_vulkan", 124,
            "vulkan_vignette_tracked_center_not_supported", 5,
            "open_vulkan_production_kernels", 2, "blocked")
        vignette_capability["semantic"]["reason"] = typed_pair(
            27, "semantic_effect_unsupported")
        vignette.update({
            "requested_capability": vignette_capability,
            "requested_backend": typed_pair(5, "vulkan"),
            "requested_provider": typed_pair(
                5, "open_vulkan_production_kernels"),
            "status": typed_pair(2, "blocked"),
            "reason": typed_pair(27, "semantic_effect_unsupported"),
            "strict_backend_selection": True,
            "strict_provider_selection": True,
            "requested": True, "blocked": True,
        })

        denoise = effects["video_noise_removal"]
        denoise_capability = typed_capability(
            1, "video_noise_removal", 1, "maxine", 2,
            "maxine.effect.unavailable", 1, "maxine_vfx", 4,
            "unavailable")
        denoise_capability["semantic"]["reason"] = typed_pair(
            28, "capability_evidence_incomplete")
        denoise.update({
            "requested_capability": denoise_capability,
            "requested_backend": typed_pair(3, "maxine"),
            "requested_provider": typed_pair(1, "maxine_vfx"),
            "status": typed_pair(2, "blocked"),
            "reason": typed_pair(28, "capability_evidence_incomplete"),
            "strict_backend_selection": True,
            "strict_provider_selection": True,
            "requested": True, "blocked": True,
        })
        key_light = effects["virtual_key_light"]
        key_light_capability = typed_capability(
            6, "virtual_key_light", 3, "open_vulkan", 117,
            "open_vulkan_matting_unavailable", 0, "", 4, "unavailable")
        key_light_capability["nested_code"] = typed_pair(
            118, "open_vulkan_matting_adapter_unavailable")
        key_light_capability["semantic"]["reason"] = typed_pair(
            27, "semantic_effect_unsupported")
        key_light.update({
            "requested_capability": key_light_capability,
            "requested_backend": typed_pair(5, "vulkan"),
            "status": typed_pair(4, "unavailable"),
            "reason": typed_pair(27, "semantic_effect_unsupported"),
            "strict_backend_selection": True,
            "requested": True, "blocked": True,
        })
        for index, lane in enumerate(value["capabilities"]["lanes"]):
            identity = (lane["effect"], lane["lane_id"])
            replacement = None
            if identity == ("vignette", 3):
                replacement = vignette_capability
            elif identity == ("video_noise_removal", 1):
                replacement = denoise_capability
            elif identity == ("virtual_key_light", 3):
                replacement = key_light_capability
            if replacement is not None:
                updated = typed_matrix_lane(
                    lane["effect_id"], lane["effect"], lane["lane_id"],
                    lane["lane"], replacement, False)
                if identity == ("virtual_key_light", 3):
                    updated["disposition"] = "experimental"
                value["capabilities"]["lanes"][index] = updated
        value["authority"].update({
            "complete": False,
            "compatibility_uncovered_effect_mask":
                (1 << 0) | (1 << 5) | (1 << 7),
        })
        parsed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": value,
        })
        self.assertFalse(parsed["valid"])
        self.assertFalse(parsed["recommendation_eligible"])
        self.assertEqual(
            parsed["reason_codes"], ["typed_status.authority_incomplete"])
        self.assertEqual(parsed["effects"], {})

    def test_typed_status_recognizes_native_cuda_provider_shadow_only(self):
        value = typed_multi_island_fixture()
        value["effects"][2] = typed_effect(3, "virtual_background.blur")
        vignette = typed_effect(8, "vignette")
        active_typed_effect(
            vignette, 2, "open_cuda", 3, "open_cuda_effect_available",
            3, 13, 2, "cuda", 7, "open_cuda_production_kernels")
        value["effects"][7] = vignette
        value["islands"][2] = typed_island(
            3, "open_cuda_virtual_background", 3, 103, 13, 8, "vignette",
            2, "cuda", 7, "open_cuda_production_kernels", 1)
        value["capabilities"]["lanes"] = typed_capability_lanes(
            value["effects"])
        value["authority"]["prepared_route_effect_mask"] = 386

        parsed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": value,
        })
        self.assertTrue(parsed["valid"], parsed["reason_codes"])
        self.assertFalse(parsed["recommendation_eligible"])
        self.assertEqual(
            parsed["islands"][2]["provider"],
            "open_cuda_production_kernels")

        mixed = typed_multi_island_fixture()
        mixed["islands"][2]["resolution"]["selected_provider"] = typed_pair(
            0, "")
        mixed["islands"][2]["readiness"]["provider"] = typed_pair(0, "")
        rejected = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": mixed,
        })
        self.assertFalse(rejected["valid"])
        self.assertFalse(rejected["recommendation_eligible"])
        self.assertEqual(
            rejected["reason_codes"],
            ["typed_status.island_selection_mismatch"])

    def test_typed_multi_island_status_accepts_maxine_zero_numeric_device_identity(self):
        value = typed_multi_island_fixture()
        maxine = value["islands"][1]
        # Production Maxine binds the stable CUDA/NvCV identity; PCI-style
        # numeric vendor/device fields may remain informational zeros.
        maxine["execution_domain"]["physical_device"]["vendor_id"] = 0
        maxine["execution_domain"]["physical_device"]["device_id"] = 0
        parsed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": value,
        })
        self.assertTrue(parsed["valid"], parsed["reason_codes"])
        self.assertEqual(
            parsed["islands"][1]["execution_domain"]["physical_device"]
            ["device_id"], 0)

        value["islands"][1]["execution_domain"]["physical_device"][
            "stable_id"] = "0" * 32
        rejected = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": value,
        })
        self.assertFalse(rejected["valid"])
        self.assertEqual(
            rejected["reason_codes"], ["typed_status.island_invalid"])

    def test_typed_multi_island_status_accepts_committed_optional_fail_open_without_active_effect(self):
        value = typed_multi_island_fixture()
        for island in value["islands"]:
            island["stages"][0]["committed_status"] = typed_pair(
                2, "optional_failed_open")
            island["latest_committed_frame"]["stages"][0]["status"] = typed_pair(
                2, "optional_failed_open")
            island["latest_committed_frame"]["stages"][0]["reason"] = typed_pair(
                12, "optional_stage_failed_open")
        for effect in value["effects"]:
            if not effect["active"]:
                continue
            effect.update({
                "active_backend": typed_pair(0, ""),
                "active_provider": typed_pair(0, ""),
                "status": typed_pair(3, "degraded"),
                "reason": typed_pair(12, "optional_stage_failed_open"),
                "committed_stage_status": typed_pair(
                    2, "optional_failed_open"),
                "active": False, "degraded": True, "fallback": True,
            })
        parsed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": value,
        })
        self.assertTrue(parsed["valid"], parsed["reason_codes"])
        self.assertTrue(parsed["writer_commit"]["valid"])
        self.assertFalse(any(
            effect["active"] for effect in parsed["effects"].values()))
        self.assertTrue(all(
            not effect["requested"] or effect["degraded"]
            for effect in parsed["effects"].values()))

    def test_typed_multi_island_status_rejects_selected_partial_capability_proof(self):
        value = typed_multi_island_fixture()
        mirror = next(
            effect for effect in value["effects"]
            if effect["effect"]["token"] == "mirror")
        mirror["selected_capability"] = copy.deepcopy(
            mirror["selected_capability"])
        mirror["selected_capability"]["semantic"]["exact_evidence"] = False
        parsed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": value,
        })
        self.assertFalse(parsed["valid"])
        self.assertEqual(
            parsed["reason_codes"],
            ["typed_status.selected_capability_incomplete"])

    def test_typed_multi_island_status_rejects_misaligned_island_runtime_bindings(self):
        cases = []

        resolution_status = typed_multi_island_fixture()
        resolution_status["islands"][0]["resolution"]["status"] = typed_pair(
            7, "failed")
        cases.append((
            resolution_status, "typed_status.island_selection_mismatch"))

        readiness_status = typed_multi_island_fixture()
        readiness_status["islands"][0]["readiness"]["status"] = typed_pair(
            11, "stale")
        cases.append((
            readiness_status, "typed_status.island_readiness_mismatch"))

        active_backend = typed_multi_island_fixture()
        active_backend["islands"][0]["resolution"]["active_backend"] = (
            typed_pair(3, "maxine"))
        cases.append((
            active_backend, "typed_status.island_selection_mismatch"))

        selected_provider = typed_multi_island_fixture()
        selected_provider["islands"][0]["resolution"]["selected_provider"] = (
            typed_pair(2, "maxine_ar"))
        cases.append((
            selected_provider, "typed_status.island_selection_mismatch"))

        program = typed_multi_island_fixture()
        program["islands"][0]["latest_committed_frame"]["program_id"] += 1
        cases.append((program, "typed_status.commit_program_mismatch"))

        invalid_commit = typed_multi_island_fixture()
        commit = invalid_commit["islands"][0]["latest_committed_frame"]
        commit["valid"] = False
        commit["capture_sequence"] = 0
        commit["output_commit_sequence"] = 0
        cases.append((invalid_commit, "typed_status.invalid_commit"))

        stage_identity = typed_multi_island_fixture()
        stage_identity["islands"][0]["resolution"]["stages"][0][
            "stage_id"] += 100
        cases.append((
            stage_identity, "typed_status.island_stage_mismatch"))

        stage_required = typed_multi_island_fixture()
        stage_required["islands"][0]["resolution"]["stages"][0][
            "required"] = True
        cases.append((
            stage_required, "typed_status.island_stage_mismatch"))

        stage_domain = typed_multi_island_fixture()
        stage_domain["islands"][0]["resolution"]["stages"][0][
            "execution_domain"] = copy.deepcopy(
                stage_domain["islands"][0]["resolution"]["stages"][0]
                ["execution_domain"])
        stage_domain["islands"][0]["resolution"]["stages"][0][
            "execution_domain"]["queue_or_stream_identity"] += 1
        cases.append((
            stage_domain,
            "typed_status.island_resolution_stage_mismatch"))

        duplicate_stage = typed_multi_island_fixture()
        duplicate_stage["islands"][0]["resolution"]["stages"].append(
            copy.deepcopy(
                duplicate_stage["islands"][0]["resolution"]["stages"][0]))
        duplicate_stage["islands"][0]["stages"].append(copy.deepcopy(
            duplicate_stage["islands"][0]["stages"][0]))
        cases.append((
            duplicate_stage,
            "typed_status.island_resolution_stage_mismatch"))

        missing_stage_record = typed_multi_island_fixture()
        missing_stage_record["islands"][0]["stages"] = []
        cases.append((
            missing_stage_record, "typed_status.island_stage_mismatch"))

        unhealthy = typed_multi_island_fixture()
        unhealthy["islands"][0]["telemetry"]["runtime_status"] = typed_pair(
            7, "failed")
        cases.append((
            unhealthy, "typed_status.active_attribution_invalid"))

        telemetry_provider = typed_multi_island_fixture()
        telemetry_provider["islands"][0]["telemetry"]["active_provider"] = (
            typed_pair(2, "maxine_ar"))
        cases.append((
            telemetry_provider, "typed_status.island_telemetry_mismatch"))

        for value, reason in cases:
            with self.subTest(reason=reason):
                parsed = parse_typed_multi_island_video_status({
                    "typed_multi_island_video_status": value,
                })
                self.assertFalse(parsed["valid"])
                self.assertEqual(parsed["reason_codes"], [reason])

    def test_untyped_and_malformed_status_never_promote_installer_effects(self):
        absent = parse_typed_multi_island_video_status({
            "engines": {"open_vulkan": {"ok": True}},
        })
        malformed = parse_typed_multi_island_video_status({
            "typed_multi_island_video_status": "legacy-ready",
        })
        self.assertEqual(absent["reason_codes"], ["typed_status.absent"])
        self.assertFalse(absent["present"] or absent["valid"])
        self.assertEqual(
            malformed["reason_codes"], ["typed_status.root_invalid"])
        self.assertTrue(malformed["present"])
        self.assertFalse(malformed["valid"])
        facts = base_facts()
        original = copy.deepcopy(facts["effects"]["capabilities"])
        facts["runtime"]["typed_multi_island_video_status"] = absent
        self.s.write_facts(facts)
        self.recommendation(facts)
        self.assertEqual(facts["effects"]["capabilities"], original)

    def test_host_analyze_exposes_typed_status_only_as_shadow_facts(self):
        current_bin = (
            Path(self.s.env["XDG_DATA_HOME"]) /
            "studiocast/current/bin")
        current_bin.mkdir(parents=True)
        ctl = current_bin / "studiocastctl"
        legacy = {
            "engines": {
                "open_vulkan": {
                    "compiled_enabled": False,
                    "blocked_reason": "vulkan_backend_disabled_in_build",
                },
            },
        }
        status = copy.deepcopy(legacy)
        status["typed_multi_island_video_status"] = (
            typed_multi_island_fixture())

        def write_status(value):
            ctl.write_text(
                "#!/bin/sh\nprintf '%s\\n' " +
                repr(json.dumps(value, separators=(",", ":"))) + "\n",
                encoding="utf-8")
            ctl.chmod(0o755)

        write_status(status)
        typed_facts = self.s.json(
            "analyze", "--target-version", "0.2.9")
        typed = typed_facts["runtime"]["typed_multi_island_video_status"]
        self.assertTrue(typed["present"] and typed["valid"])
        self.assertEqual(typed["mode"], "shadow_only")
        self.assertFalse(typed["recommendation_eligible"])

        write_status(legacy)
        legacy_facts = self.s.json(
            "analyze", "--target-version", "0.2.9")
        self.assertFalse(
            legacy_facts["runtime"]["typed_multi_island_video_status"]
            ["present"])
        self.assertEqual(
            typed_facts["effects"]["capabilities"],
            legacy_facts["effects"]["capabilities"])

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

    def test_helper_analysis_uses_full_fake_trust_and_transport_preflight(self):
        root = self.s.root / "preflight"
        root.mkdir()
        helper, pkexec = root / "helper", root / "pkexec"
        for path in (helper, pkexec):
            path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            path.chmod(0o755)
        policy = root / "policy.xml"
        policy.write_text(f"""<policyconfig><action id="com.studiocast.Installer1.manage-host-integration">
<annotate key="org.freedesktop.policykit.exec.path">{helper}</annotate>
<annotate key="org.freedesktop.policykit.exec.allow_gui">true</annotate>
</action></policyconfig>""", encoding="utf-8")
        policy.chmod(0o644)
        service = root / "polkit.service"
        service.write_text("[D-BUS Service]\nName=org.freedesktop.PolicyKit1\n", encoding="utf-8")
        service.chmod(0o644)
        bus_path = root / "system_bus_socket"
        bus = socket.socket(socket.AF_UNIX)
        bus.bind(str(bus_path))
        env = self.s.env | {
            "STUDIOCAST_INSTALLER_TEST_HELPER": str(helper),
            "STUDIOCAST_INSTALLER_TEST_PKEXEC": str(pkexec),
            "STUDIOCAST_INSTALLER_TEST_POLKIT_POLICY": str(policy),
            "STUDIOCAST_INSTALLER_TEST_SYSTEM_BUS": str(bus_path),
            "STUDIOCAST_INSTALLER_TEST_POLKIT_SERVICE": str(service),
        }
        try:
            facts = self.s.json("analyze", "--target-version", "0.2.9", env=env)
        finally:
            bus.close()
            bus_path.unlink()
        self.assertEqual(facts["privileged_helper"]["reason_codes"],
                         ["privileged_helper_usable"])
        self.assertTrue(facts["privileged_helper"]["trusted"])
        self.assertTrue(facts["privileged_helper"]["authorization_transport_available"])
        unavailable = self.s.json("analyze", "--target-version", "0.2.9", env=env)
        self.assertEqual(unavailable["privileged_helper"]["reason_codes"],
                         ["authorization_transport_unavailable"])

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
        facts["cache"]["release_artifacts"] = {
            "studiocast-source": {"verified": True, "state": "cached_verified"}}
        self.assertEqual(self.recommendation(facts)["route"], "recommended")

    def test_recommendation_is_deterministic_models_exact_and_legacy_vulkan_fails_closed(self):
        one = self.recommendation(); two = self.recommendation()
        self.assertEqual(one, two)
        self.assertEqual(one["selections"]["model_pack_ids"], DEFAULT_PACKS)
        self.assertEqual(one["disk"], {"free_bytes": 20_000_000_000,
                                       "base_required_bytes": 3,
                                       "download_bytes": 131224287,
                                       "required_bytes": 131224290})
        self.assertNotIn("artifact_sizes_untrusted", {item["code"] for item in one["blockers"]})
        self.assertEqual(one["selections"]["effects"]["virtual_background"]["selected"], "cuda")
        facts = base_facts(); facts["effects"]["capabilities"]["virtual_background"]["cuda"] = "unavailable"
        # A legacy string or forged aggregate availability has no exact
        # per-effect production evidence and therefore cannot select Vulkan.
        self.assertEqual(self.recommendation(facts)["selections"]["effects"]["virtual_background"]["selected"], "unavailable")
        facts["maxine"] = {"components": {"video_fx": {"usable": True}}, "effects": {}, "reason_codes": []}
        facts["effects"]["capabilities"]["virtual_background"]["maxine"] = "production_usable"
        self.assertEqual(self.recommendation(facts)["selections"]["effects"]["virtual_background"]["selected"], "maxine")
        self.assertEqual(self.recommendation(facts)["selections"]["model_pack_ids"], DEFAULT_PACKS)

    def test_recommendation_never_reads_replaced_packaged_catalog(self):
        component = self.s.root / "pure-recommend-component"
        installer_root = component / "installer"
        staged_backend = stage_backend(installer_root)
        model_dir = installer_root / "models"
        model_dir.mkdir()
        for name in ("__init__.py", "model_transactions.py"):
            shutil.copy2(SOURCE / "installer/models" / name, model_dir / name)
        catalog_path = component / "installer/models/curated-model-catalog-v1.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / "installer/models/curated-model-catalog-v1.json", catalog_path)
        facts_path = self.s.root / "pure-recommend-facts.json"
        facts_path.write_text(json.dumps(base_facts()), encoding="utf-8")

        def invoke() -> bytes:
            result = subprocess.run(
                [str(staged_backend), "recommend", "--facts", str(facts_path),
                 "--intent", "install"], env=self.s.env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            return result.stdout

        before = invoke()
        changed = json.loads(catalog_path.read_text(encoding="utf-8"))
        changed["packs"][0]["artifacts"][0]["size_bytes"] += 1
        catalog_path.write_text(json.dumps(changed), encoding="utf-8")
        after = invoke()
        self.assertEqual(before, after)
        plan = subprocess.run(
            [str(staged_backend), "plan", "install", "--facts", str(facts_path),
             "--no-v4l2loopback", "--no-service", "--models"], env=self.s.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("artifact_sizes_untrusted",
                      {item["code"] for item in json.loads(plan.stdout)["blockers"]})

    def test_recommendation_rejects_incomplete_or_inconsistent_size_facts(self):
        cases = {}
        missing = base_facts()
        del missing["models"]["catalog_size_summary"]
        cases["missing_summary"] = missing
        wrong_bool = base_facts()
        wrong_bool["models"]["catalog_size_summary"]["download_bytes"] = True
        cases["boolean_total"] = wrong_bool
        nonpositive = base_facts()
        nonpositive["models"]["catalog_size_summary"]["packs"][0]["download_bytes"] = 0
        cases["nonpositive_pack_total"] = nonpositive
        wrong_status = base_facts()
        wrong_status["models"]["catalog_size_summary"]["packs"][0]["size_status"] = "unknown"
        cases["status_inconsistent"] = wrong_status
        duplicate = base_facts()
        duplicate["models"]["catalog_size_summary"]["packs"][-1]["id"] = DEFAULT_PACKS[0]
        cases["duplicate_pack_id"] = duplicate
        missing_pack = base_facts()
        missing_pack["models"]["catalog_size_summary"]["packs"].pop()
        cases["missing_pack_id"] = missing_pack
        inconsistent = base_facts()
        inconsistent["models"]["catalog_size_summary"]["download_bytes"] += 1
        cases["inconsistent_total"] = inconsistent
        duplicate_defaults = base_facts()
        duplicate_defaults["models"]["default_pack_ids"][-1] = DEFAULT_PACKS[0]
        cases["duplicate_default_id"] = duplicate_defaults
        untrusted = base_facts()
        untrusted["models"]["catalog_size_summary"]["trusted"] = False
        cases["untrusted"] = untrusted

        for name, facts in cases.items():
            with self.subTest(name=name):
                rec = self.recommendation(facts)
                self.assertIn("artifact_sizes_untrusted",
                              {item["code"] for item in rec["blockers"]})
                self.assertEqual(rec["disk"]["download_bytes"], 0)

    def test_recommended_open_vulkan_feature_exactly_tracks_selected_effects(self):
        zero = base_facts(desired_configuration={
            "features": {"open_vulkan": True}})
        zero["effects"]["capabilities"] = {}
        zero_rec = self.recommendation(zero)
        self.assertEqual(zero_rec["route"], "recommended")
        self.assertFalse(zero_rec["selections"]["features"]["open_vulkan"])
        self.assertFalse(any(
            item["selected"] == "vulkan"
            for item in zero_rec["selections"]["effects"].values()))

        for retained in ("mirror", "vignette"):
            with self.subTest(retained=retained):
                facts = machine_recommendation_facts("amd")
                facts["installation"]["desired_configuration"] = {
                    "features": {"open_vulkan": False}}
                removed = "vignette" if retained == "mirror" else "mirror"
                matrix = facts["effects"]["capabilities"][removed]
                matrix["vulkan"] = "unavailable"
                matrix["vulkan_evidence"]["production_ready"] = False
                matrix["success_reason_codes"]["vulkan"] = []
                matrix["blocker_reason_codes"]["vulkan"] = [
                    f"open_vulkan_{removed}_unavailable"]
                rec = self.recommendation(facts)
                selected = rec["selections"]["effects"]
                self.assertEqual(
                    [effect for effect, item in selected.items()
                     if item["selected"] == "vulkan"],
                    [retained])
                self.assertTrue(rec["selections"]["features"]["open_vulkan"])

    def test_non_recommended_routes_preserve_prior_open_vulkan_intent(self):
        modify = base_facts(
            classification="healthy", active_version="0.2.9",
            version_relation="same", desired_configuration={
                "features": {"open_vulkan": True}})
        modify["effects"]["capabilities"] = {}
        modify_rec = self.recommendation(modify)
        self.assertEqual(modify_rec["route"], "modify")
        self.assertTrue(modify_rec["selections"]["features"]["open_vulkan"])

        valid = machine_recommendation_facts("amd")
        valid["installation"]["desired_configuration"] = {
            "features": {"open_vulkan": False}}
        for route in ("advanced", "custom"):
            with self.subTest(route=route):
                rec = self.recommendation(valid, "--route", route)
                self.assertEqual(rec["route"], route)
                self.assertFalse(rec["selections"]["features"]["open_vulkan"])
                self.assertTrue(any(
                    item["selected"] == "vulkan"
                    for item in rec["selections"]["effects"].values()))

    def test_missing_and_partial_capability_maps_emit_all_canonical_effects(self):
        for capabilities in ({}, []):
            with self.subTest(capabilities=type(capabilities).__name__):
                facts = base_facts()
                facts["effects"]["capabilities"] = capabilities
                selected = self.recommendation(facts)["selections"]["effects"]
                self.assertEqual(list(selected), sorted(CANONICAL_EFFECTS))
                for effect, item in selected.items():
                    self.assertEqual(item["selected"], "unavailable")
                    expected = {
                        engine: [f"effect.{effect}.{engine}.capability_evidence_missing"]
                        for engine in ("maxine", "cuda", "vulkan", "cpu")
                    }
                    self.assertEqual(
                        item["capability_blocker_reason_codes"], expected)
                    self.assertEqual(
                        item["selected_reason_codes"],
                        [codes[0] for codes in expected.values()])

        partial = machine_recommendation_facts("amd")
        partial["effects"]["capabilities"] = {
            "mirror": partial["effects"]["capabilities"]["mirror"]}
        rec = self.recommendation(partial)
        selected = rec["selections"]["effects"]
        self.assertEqual(list(selected), sorted(CANONICAL_EFFECTS))
        self.assertEqual(selected["mirror"]["selected"], "vulkan")
        self.assertTrue(rec["selections"]["features"]["open_vulkan"])
        for effect in CANONICAL_EFFECTS[1:]:
            self.assertEqual(selected[effect]["selected"], "unavailable")
            self.assertEqual(
                selected[effect]["capability_blocker_reason_codes"]["vulkan"],
                [f"effect.{effect}.vulkan.capability_evidence_missing"])

    def test_inconsistent_vulkan_attestation_has_stable_fallback_blocker(self):
        mutations = (
            ("success_missing", lambda matrix: matrix["success_reason_codes"].update(vulkan=[])),
            ("success_wrong", lambda matrix: matrix["success_reason_codes"].update(
                vulkan=["open_vulkan_wrong_production_ready"])),
            ("common_evidence_missing", lambda matrix: matrix["vulkan_evidence"].pop(
                "runtime_library_found")),
            ("effect_evidence_missing", lambda matrix: matrix["vulkan_evidence"].pop(
                "model_artifact_ready")),
        )
        for name, mutate in mutations:
            with self.subTest(mutation=name):
                facts = machine_recommendation_facts("amd")
                matrix = promote_production_looking_vulkan(
                    facts, "virtual_background.blur")
                mutate(matrix)
                item = self.recommendation(facts)["selections"]["effects"][
                    "virtual_background.blur"]
                reason = ("effect.virtual_background.blur.vulkan."
                          "production_evidence_inconsistent")
                self.assertNotEqual(item["selected"], "vulkan")
                self.assertEqual(
                    item["capability_blocker_reason_codes"]["vulkan"],
                    [reason])
                self.assertIn(reason, item["selected_reason_codes"])

        facts = machine_recommendation_facts("amd")
        matrix = promote_production_looking_vulkan(
            facts, "virtual_background.blur")
        matrix["vulkan_evidence"].pop("runtime_ready")
        matrix["blocker_reason_codes"]["vulkan"] = [
            "open_vulkan_matting_runtime_failure"]
        item = self.recommendation(facts)["selections"]["effects"][
            "virtual_background.blur"]
        self.assertEqual(
            item["capability_blocker_reason_codes"]["vulkan"],
            ["open_vulkan_matting_runtime_failure"])

    def test_per_effect_recommendation_fixtures_cover_gpu_topologies_and_failures(self):
        valid_cases = (("nvidia", "valid"), ("amd", "valid"),
                       ("intel", "valid"), ("hybrid", "valid"))
        for vendor, profile in valid_cases:
            with self.subTest(vendor=vendor, profile=profile):
                rec = self.recommendation(machine_recommendation_facts(vendor, profile))
                selected = rec["selections"]["effects"]
                self.assertEqual(selected["mirror"]["selected"], "vulkan")
                self.assertEqual(selected["vignette"]["selected"], "vulkan")
                self.assertEqual(
                    selected["vignette"]["selected_reason_codes"],
                    ["open_vulkan_vignette_fixed_center_production_ready"])
                self.assertEqual(
                    machine_recommendation_facts(vendor, profile)["effects"]
                    ["capabilities"]["vignette"]["vulkan_evidence"]
                    ["parameter_contract"], "fixed_center")
                self.assertTrue(rec["selections"]["features"]["open_vulkan"])
                for effect in (
                    "virtual_background.blur", "virtual_background.remove",
                    "virtual_background.replace", "auto_frame", "eye_contact",
                    "video_noise_removal", "virtual_key_light",
                ):
                    self.assertNotEqual(selected[effect]["selected"], "vulkan")
                self.assertTrue(all(item["selected"] != "cpu"
                                    for item in selected.values()))

        failure_cases = (
            ("none", "no_gpu", "vulkan_no_physical_device"),
            ("none", "loader_only", "vulkan_no_physical_device"),
            ("none", "cpu_only", "vulkan_only_cpu_devices_available"),
            ("amd", "no_compute", "vulkan_no_compute_queue"),
            ("nvidia", "device_lost", "vulkan_device_lost"),
        )
        for vendor, profile, blocker in failure_cases:
            with self.subTest(vendor=vendor, profile=profile):
                rec = self.recommendation(
                    machine_recommendation_facts(vendor, profile, blocker))
                selected = rec["selections"]["effects"]
                self.assertTrue(all(item["selected"] != "vulkan"
                                    for item in selected.values()))
                self.assertIn(
                    blocker,
                    selected["mirror"]["capability_blocker_reason_codes"]
                    ["vulkan"])
                self.assertTrue(all(item["selected"] != "cpu"
                                    for item in selected.values()))

    def test_effect_precedence_isolation_fixed_center_and_cpu_contract(self):
        facts = machine_recommendation_facts("amd")
        mirror = facts["effects"]["capabilities"]["mirror"]
        mirror["cuda"] = "production_usable"
        mirror["success_reason_codes"]["cuda"] = ["open_cuda_effect_available"]
        self.assertEqual(self.recommendation(facts)["selections"]["effects"]
                         ["mirror"]["selected"], "cuda")
        mirror["maxine"] = "production_usable"
        mirror["success_reason_codes"]["maxine"] = ["maxine.effect.usable"]
        self.assertEqual(self.recommendation(facts)["selections"]["effects"]
                         ["mirror"]["selected"], "maxine")

        isolated = machine_recommendation_facts("intel")
        isolated_mirror = isolated["effects"]["capabilities"]["mirror"]
        isolated_mirror["vulkan"] = "unavailable"
        isolated_mirror["vulkan_evidence"]["production_ready"] = False
        isolated_mirror["success_reason_codes"]["vulkan"] = []
        isolated_mirror["blocker_reason_codes"]["vulkan"] = ["vulkan_device_lost"]
        rec = self.recommendation(isolated)
        self.assertEqual(rec["selections"]["effects"]["mirror"]["selected"],
                         "unavailable")
        self.assertEqual(rec["selections"]["effects"]["vignette"]["selected"],
                         "vulkan")

        tracked = machine_recommendation_facts("nvidia")
        vignette = tracked["effects"]["capabilities"]["vignette"]
        vignette["vulkan"] = "unavailable"
        vignette["vulkan_evidence"].update(
            production_ready=False, parameter_contract="tracked_center")
        vignette["success_reason_codes"]["vulkan"] = []
        vignette["blocker_reason_codes"]["vulkan"] = [
            "vulkan_vignette_tracked_center_not_supported"]
        tracked_rec = self.recommendation(tracked)
        self.assertNotEqual(tracked_rec["selections"]["effects"]["vignette"]
                            ["selected"], "vulkan")
        self.assertIn(
            "vulkan_vignette_tracked_center_not_supported",
            tracked_rec["selections"]["effects"]["vignette"]
            ["selected_reason_codes"])

        auto = tracked["effects"]["capabilities"]["auto_frame"]
        auto["vulkan"] = "production_usable"
        auto["vulkan_evidence"]["production_ready"] = False
        self.assertNotEqual(self.recommendation(tracked)["selections"]["effects"]
                            ["auto_frame"]["selected"], "vulkan")
        mirror = tracked["effects"]["capabilities"]["mirror"]
        mirror["maxine"] = "unavailable"
        mirror["cuda"] = "unavailable"
        mirror["vulkan"] = "unavailable"
        mirror["cpu"] = "production_usable"
        self.assertEqual(self.recommendation(tracked)["selections"]["effects"]
                         ["mirror"]["selected"], "unavailable")
        for effect, matrix in tracked["effects"]["capabilities"].items():
            if effect != "mirror":
                self.assertEqual(matrix["cpu"], "unavailable")
            self.assertFalse(matrix["cpu_evidence"]["selectable_production_path"])
            self.assertEqual(
                matrix["blocker_reason_codes"]["cpu"],
                [f"effect.{effect}.cpu.no_selectable_production_path"])

    def test_effect_specific_vulkan_evidence_cannot_be_forged_by_composites(self):
        adversarial_cases = (
            ("virtual_background.blur", "runtime_ready", "clear"),
            ("virtual_background.blur", "no_cpu_layers", "omit"),
            ("virtual_background.blur", "no_cpu_readback", "clear"),
            ("virtual_background.blur", "no_dynamic_allocations", "omit"),
            ("eye_contact", "model_artifact_ready", "omit"),
            ("eye_contact", "synchronization_contract_validated", "clear"),
            ("eye_contact", "bounded_reusable_allocations", "omit"),
            ("eye_contact", "parity_validated", "clear"),
            ("eye_contact", "no_cpu_fallback", "omit"),
            ("video_noise_removal", "device_resident", "clear"),
            ("video_noise_removal", "no_cpu_readback", "omit"),
            ("video_noise_removal", "temporal_history_bounded", "clear"),
            ("video_noise_removal", "history_reset_on_reconfigure", "omit"),
            ("video_noise_removal", "capture_sequence_discontinuity_reset", "clear"),
        )
        proven_effects = {effect for effect, _, _ in adversarial_cases}
        for effect in proven_effects:
            with self.subTest(effect=effect, proof="complete"):
                facts = machine_recommendation_facts("amd")
                promote_production_looking_vulkan(facts, effect)
                self.assertEqual(
                    self.recommendation(facts)["selections"]["effects"][effect]
                    ["selected"], "vulkan")

        for effect, key, mutation in adversarial_cases:
            with self.subTest(effect=effect, evidence=key, mutation=mutation):
                facts = machine_recommendation_facts("amd")
                matrix = promote_production_looking_vulkan(facts, effect)
                if mutation == "omit":
                    matrix["vulkan_evidence"].pop(key)
                else:
                    matrix["vulkan_evidence"][key] = False
                self.assertNotEqual(
                    self.recommendation(facts)["selections"]["effects"][effect]
                    ["selected"], "vulkan")

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

    def test_dependency_load_and_persistence_switches_change_exact_operations(self):
        facts = base_facts()
        facts["toolchain"]["missing_build_dependencies"] = ["cmake", "c++", "pkg-config"]
        facts["v4l2"].update(module_available=True, module_loaded=False, device_present=False,
                              persistence_state="absent")
        facts["privileged_helper"].update(present=True, trusted=True, compatible=True,
                                           authorization_transport_available=True)
        self.s.write_facts(facts)
        managed = self.s.json("plan", "advanced", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--with-deps", "--v4l2loopback", "--load-loopback",
            "--no-persist-loopback", "--no-service", "--no-models", "--allow-unsupported")
        packages = next(op for op in managed["operations"]
                        if op["id"] == "dependencies.packages.ensure")
        self.assertEqual(packages["inputs"]["packages"],
                         ["cmake", "build-essential", "pkg-config"])
        ids = [op["id"] for op in managed["operations"]]
        self.assertLess(ids.index("dependencies.packages.ensure"), ids.index("build.configure"))
        self.assertIn("v4l.module.load", ids)
        self.assertNotIn("v4l.persistence.write", ids)

        skipped = self.s.json("plan", "advanced", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--skip-deps", "--v4l2loopback", "--no-load-loopback",
            "--persist-loopback", "--no-service", "--no-models", "--allow-unsupported")
        ids = [op["id"] for op in skipped["operations"]]
        self.assertNotIn("dependencies.packages.ensure", ids)
        self.assertNotIn("v4l.module.load", ids)
        self.assertIn("v4l.persistence.write", ids)
        self.assertIn("toolchain.build_dependencies_missing",
                      [item["code"] for item in skipped["blockers"]])

    def test_apply_rejects_tamper_and_exact_dry_run_executes(self):
        plan = self.plan()
        path = self.s.root / "plan.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], "--facts", str(self.s.facts), "--dry-run")
        value = terminal_result(result.stdout)
        self.assertEqual(value["reviewed_operation_ids"], value["executed_operation_ids"])
        events = progress_events(result.stdout)
        self.assertEqual(len(events), len(plan["operations"]) * 2)
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual([(event["operation_id"], event["boundary"]) for event in events],
                         [(op["id"], boundary) for op in plan["operations"]
                          for boundary in ("before", "after")])
        self.assertTrue(all(event["plan_id"] == plan["plan_id"] and
                            event["plan_digest"] == plan["plan_digest"] and
                            event["transaction_id"] == value["transaction_id"] for event in events))
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

    def test_migrated_unknown_service_and_v4l_require_explicit_selection(self):
        facts = base_facts(classification="partial", active_version="0.2.8",
            desired_configuration={"build_type": "unknown", "features": {},
                "service": {"desired": "unknown"}, "v4l": {"desired": "unknown"},
                "model_pack_ids": [], "preserve_settings": True})
        self.s.write_facts(facts)
        blocked = self.s.json("plan", "repair", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-models", "--allow-unsupported")
        codes = [item["code"] for item in blocked["blockers"]]
        self.assertIn("configuration.service.selection_required", codes)
        self.assertIn("configuration.virtual_camera.selection_required", codes)
        ids = [op["id"] for op in blocked["operations"]]
        self.assertFalse(any(op_id.startswith("v4l.") for op_id in ids))
        self.assertFalse(any(op_id.startswith("service.") for op_id in ids))

        selected = self.s.json("plan", "repair", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-models", "--no-v4l2loopback",
            "--no-service", "--allow-unsupported")
        selected_codes = [item["code"] for item in selected["blockers"]]
        self.assertNotIn("configuration.service.selection_required", selected_codes)
        self.assertNotIn("configuration.virtual_camera.selection_required", selected_codes)

    def test_pack_catalog_is_exact_seven_ids_eight_hashed_artifacts(self):
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts), "--source-dir", str(SOURCE),
                           "--no-v4l2loopback", "--no-service", "--models", "--allow-unsupported")
        downloads = plan["downloads"]
        self.assertEqual(len(downloads), 8)
        self.assertEqual({x["pack_id"] for x in downloads}, set(DEFAULT_PACKS))
        self.assertTrue(all(x["sha256"].startswith("sha256:") for x in downloads))
        self.assertEqual([x["size"] for x in downloads],
                         [832775, 2033628, 25888640, 232589, 99693937,
                          1062936, 1062994, 416788])
        self.assertTrue(all(x["size_status"] == "known" for x in downloads))
        self.assertEqual(plan["disk"]["download_bytes"], 131224287)
        self.assertEqual(plan["disk"]["required_bytes"], 131224290)
        self.assertEqual(plan["disk"]["free_bytes"], 20_000_000_000)
        transactions = [x for x in plan["operations"] if x["kind"] == "model_transaction"]
        self.assertEqual(len(transactions), 7)
        gaze = next(x for x in transactions if x["inputs"]["pack_id"] == "gaze_correction_cam_flx_v0_1_1")
        self.assertEqual([a["name"] for a in gaze["inputs"]["pack"]["artifacts"]],
                         ["gaze_flx_left.onnx", "gaze_flx_right.onnx"])

    def test_recommended_models_use_trusted_sizes_and_low_disk_blocks_before_apply(self):
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts),
                           "--no-v4l2loopback", "--no-service", "--models")
        self.assertNotIn("artifact_sizes_untrusted", [item["code"] for item in plan["blockers"]])
        transactions = [op for op in plan["operations"] if op["kind"] == "model_transaction"]
        self.assertTrue(transactions)
        self.assertTrue(all(op["inputs"]["recommended"] for op in transactions))
        advanced = self.plan("--models")
        self.assertTrue(all(not op["inputs"]["recommended"] for op in advanced["operations"]
                            if op["kind"] == "model_transaction"))

        low_disk_facts = base_facts()
        low_disk_facts["paths"]["free_bytes"] = 131224289
        self.s.write_facts(low_disk_facts)
        blocked = self.s.json("plan", "install", "--facts", str(self.s.facts),
                              "--no-v4l2loopback", "--no-service", "--models")
        disk_blocker = next(item for item in blocked["blockers"]
                            if item["code"] == "disk.staging.insufficient")
        self.assertEqual((disk_blocker["free_bytes"], disk_blocker["required_bytes"]),
                         (131224289, 131224290))
        path = self.s.root / "blocked-model-plan.json"
        path.write_text(json.dumps(blocked), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path),
                            "--digest", blocked["plan_digest"],
                            "--token", blocked["approval_token"],
                            "--facts", str(self.s.facts), check=False)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((Path(self.s.env["XDG_CACHE_HOME"]) / "studiocast/builds").exists())

        repair_facts = base_facts(classification="partial", active_version="0.2.8",
            desired_configuration={"build_type": "Release", "features": {},
                "service": {"desired": "disabled"},
                "v4l": {"desired": False, "required_for_success": False},
                "model_pack_ids": list(DEFAULT_PACKS), "preserve_settings": True})
        self.s.write_facts(repair_facts)
        repair = self.s.json("plan", "repair", "--facts", str(self.s.facts),
                             "--no-v4l2loopback", "--no-service", "--models")
        self.assertEqual((repair["route"], repair["detected_route"]),
                         ("recommended", "repair"))
        self.assertNotIn("artifact_sizes_untrusted", [item["code"] for item in repair["blockers"]])
        self.assertTrue(all(op["inputs"]["recommended"] for op in repair["operations"]
                            if op["kind"] == "model_transaction"))

    def test_backend_executes_packaged_atomic_model_transaction_and_degrades_on_missing_source(self):
        component = self.s.root / "component"
        installer_root = component / "installer"
        staged_backend = stage_backend(installer_root)
        model_dir = installer_root / "models"; model_dir.mkdir()
        for name in ("__init__.py", "model_transactions.py"):
            shutil.copy2(SOURCE / "installer/models" / name, model_dir / name)
        metadata_root = component / "resources/model_packs/fixture"
        metadata_root.mkdir(parents=True)
        model_json = b'{"id":"fastenhancer_s_vd_v1"}\n'
        license_text = b"fixture license\n"
        (metadata_root / "model.json").write_bytes(model_json)
        (metadata_root / "LICENSE.txt").write_bytes(license_text)
        transport_root = self.s.root / "model-source"
        artifact_path = transport_root / "objects/model.onnx"
        artifact_path.parent.mkdir(parents=True)
        artifact = b"backend model transaction fixture\n"
        artifact_path.write_bytes(artifact)
        sha = lambda value: hashlib.sha256(value).hexdigest()
        catalog = {"schema_version": 1, "catalog_version": "fixture-v1", "policy_version": 1,
            "default_source": str(transport_root),
            "size_metadata": {"trusted": True, "reason_code": "fixture_sizes"},
            "packs": [{"id": "fastenhancer_s_vd_v1", "family": "open_audio",
                "task": "audio_enhancement", "default": True, "pack_path": "Fixture Pack",
                "metadata": [
                    {"source": "resources/model_packs/fixture/model.json", "destination": "model.json", "sha256": sha(model_json)},
                    {"source": "resources/model_packs/fixture/LICENSE.txt", "destination": "LICENSE.txt", "sha256": sha(license_text)}],
                "license": {"spdx": "MIT"}, "provenance": {"project": "fixture"},
                "artifacts": [{"name": "model.onnx", "sha256": sha(artifact),
                    "size_bytes": len(artifact), "size_status": "known",
                    "source_paths": ["objects/model.onnx"]}]}]}
        catalog_path = component / "installer/models/curated-model-catalog-v1.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(catalog))

        def run_for(root: Path, *, expect_success: bool,
                    tamper_catalog: bool = False) -> tuple[subprocess.CompletedProcess[str], dict]:
            catalog_path.write_text(json.dumps(catalog))
            env = self.s.env | {"HOME": str(root / "home"), "XDG_DATA_HOME": str(root / "data"),
                "XDG_CONFIG_HOME": str(root / "config"), "XDG_STATE_HOME": str(root / "state"),
                "XDG_CACHE_HOME": str(root / "cache"), "STUDIOCAST_INSTALLER_TEST_MODE": "1",
                "STUDIOCAST_INSTALLER_OFFLINE": "0"}
            facts_path = root / "facts.json"; root.mkdir(parents=True, exist_ok=True)
            facts_path.write_text(json.dumps(base_facts()))
            custom = root / "data/studiocast/models/open_audio/custom-pack"
            custom.mkdir(parents=True); (custom / "keep.txt").write_text("keep")
            plan_process = subprocess.run([str(staged_backend), "plan", "advanced", "--facts", str(facts_path),
                "--source-dir", str(SOURCE), "--no-v4l2loopback", "--no-service",
                "--model-id", "fastenhancer_s_vd_v1", "--model-source", str(transport_root),
                "--allow-unsupported"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(plan_process.returncode, 0, plan_process.stderr)
            plan = json.loads(plan_process.stdout)
            plan_path = root / "plan.json"; plan_path.write_text(json.dumps(plan))
            if tamper_catalog:
                changed = json.loads(catalog_path.read_text())
                changed["packs"][0]["artifacts"][0]["size_bytes"] += 1
                catalog_path.write_text(json.dumps(changed))
            fake_tools = self._fake_cmake()
            fake_env = env | {"PATH": fake_tools["PATH"]}
            result = subprocess.run([str(staged_backend), "apply-plan", "--plan", str(plan_path),
                "--digest", plan["plan_digest"], "--token", plan["approval_token"],
                "--facts", str(facts_path)], env=fake_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            value = terminal_result(result.stdout)
            self.assertTrue((custom / "keep.txt").is_file())
            if expect_success:
                self.assertEqual((result.returncode, value["state"]), (0, "committed"),
                                 (result.stderr, value))
                pack = root / "data/studiocast/models/open_audio/Fixture Pack"
                self.assertEqual((pack / "model.onnx").read_bytes(), artifact)
                self.assertTrue((pack / "model.json").is_file() and (pack / "LICENSE.txt").is_file())
                self.assertTrue((pack / ".studiocast-curated-pack.json").is_file())
            return result, value

        installed, value = run_for(self.s.root / "model-success", expect_success=True)
        self.assertEqual((installed.returncode, value["state"]), (0, "committed"))
        tampered, value = run_for(self.s.root / "model-catalog-tampered",
                                  expect_success=False, tamper_catalog=True)
        self.assertEqual(tampered.returncode, 2)
        self.assertEqual(value["error"]["code"], "model.catalog.changed")
        self.assertFalse((self.s.root / "model-catalog-tampered/data/studiocast/payloads").exists())
        self.assertFalse((self.s.root / "model-catalog-tampered/cache/studiocast/builds").exists())
        artifact_path.unlink()
        degraded, value = run_for(self.s.root / "model-missing", expect_success=False)
        self.assertEqual((degraded.returncode, value["state"], value["core_committed"]),
                         (3, "degraded", True))
        self.assertEqual(value["retryable_operation_ids"],
                         ["model.pack.fastenhancer_s_vd_v1.transaction"])

    def test_packaged_style_uninstall_is_self_contained_and_preserves_data(self):
        data = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"
        payload = data / "payloads/0.2.9/bin"; payload.mkdir(parents=True)
        current = data / "current"; current.symlink_to(data / "payloads/0.2.9")
        models = data / "models/custom"; models.mkdir(parents=True); (models / "mine").write_text("keep")
        config = Path(self.s.env["XDG_CONFIG_HOME"]) / "studiocast"; config.mkdir(parents=True); (config / "daemon.conf").write_text("keep")
        state = Path(self.s.env["XDG_STATE_HOME"]) / "studiocast"; state.mkdir(parents=True); (state / "installer.log").write_text("keep")
        cache = Path(self.s.env["XDG_CACHE_HOME"]) / "studiocast"; cache.mkdir(parents=True); (cache / "verified.cache").write_text("keep")
        plan = self.s.json("plan", "uninstall", "--facts", str(self.s.facts), "--preserve-user-data")
        path = self.s.root / "uninstall.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"], "--token", plan["approval_token"])
        self.assertEqual(terminal_result(result.stdout)["state"], "committed")
        self.assertFalse((data / "payloads").exists()); self.assertTrue((models / "mine").exists()); self.assertTrue((config / "daemon.conf").exists())
        self.assertTrue((state / "installer.log").is_file()); self.assertTrue((cache / "verified.cache").is_file())
        self.assertEqual(json.loads((data / "install-manifest.json").read_text())["transaction"]["state"], "uninstalled")

    def test_uninstall_remove_user_data_wipes_only_fixed_xdg_roots(self):
        data = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"
        config = Path(self.s.env["XDG_CONFIG_HOME"]) / "studiocast"
        state = Path(self.s.env["XDG_STATE_HOME"]) / "studiocast"
        cache = Path(self.s.env["XDG_CACHE_HOME"]) / "studiocast"
        for root in (data / "models/custom", config, state, cache):
            root.mkdir(parents=True, exist_ok=True)
            (root / "owned").write_text("remove", encoding="utf-8")
        outside = self.s.root / "outside"
        outside.write_text("keep", encoding="utf-8")
        plan = self.s.json("plan", "uninstall", "--facts", str(self.s.facts),
                           "--remove-user-data")
        cleanup = {op["inputs"].get("target") for op in plan["operations"]
                   if op["kind"] == "cleanup"}
        self.assertTrue({"models", "settings", "runtime_state", "cache", "data"} <= cleanup)
        path = self.s.root / "full-uninstall.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], "--facts", str(self.s.facts))
        self.assertEqual(terminal_result(result.stdout)["state"], "committed")
        self.assertFalse(config.exists()); self.assertFalse(state.exists()); self.assertFalse(cache.exists())
        self.assertFalse((data / "models").exists()); self.assertTrue(outside.is_file())

    def test_uninstall_remains_available_with_unsupported_override_flag(self):
        facts = base_facts()
        facts["os"]["supported"] = False
        facts["os"]["reason_codes"] = ["os.unsupported.distribution"]
        self.s.write_facts(facts)

        plan = self.s.json("plan", "uninstall", "--facts", str(self.s.facts),
                           "--no-v4l2loopback", "--no-service", "--no-models",
                           "--allow-unsupported", "--preserve-user-data")

        self.assertEqual(plan["blockers"], [])
        self.assertEqual(plan["intent"], "uninstall")

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
        value = terminal_result(result.stdout)
        self.assertEqual(value["reviewed_operation_ids"], value["executed_operation_ids"])

    def test_host_analyzer_consumes_runtime_diagnostics_and_real_vulkan_compute_evidence(self):
        current_bin = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/current/bin"
        current_bin.mkdir(parents=True)
        ctl_calls = self.s.root / "studiocastctl-calls"
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
            "open_audio": {}, "open_vulkan": {
                "compiled_enabled": True, "ok": True,
                "runtime_library_found": True, "instance_created": True,
                "physical_device_found": True, "non_cpu_device_selected": True,
                "compute_queue_available": True, "logical_device_created": True,
                "context_created": True, "context_healthy": True,
                "production_hardware_ready": True, "shader_pipeline_created": True,
                "available_effects": ["mirror", "auto_frame", "vignette"],
                "blocked_effects": {
                    "virtual_background.blur": "open_vulkan_matting_unavailable",
                    "virtual_background.remove": "open_vulkan_matting_unavailable",
                    "virtual_background.replace": "open_vulkan_matting_unavailable",
                    "virtual_key_light": "open_vulkan_matting_unavailable",
                    "eye_contact": "open_vulkan_eye_contact_unavailable",
                    "video_noise_removal": "open_vulkan_video_noise_removal_unavailable",
                },
                "mirror_production_ready": True,
                "mirror_readiness_code": "open_vulkan_mirror_production_ready",
                "vignette_fixed_center_production_ready": True,
                "vignette_readiness_code":
                    "open_vulkan_vignette_fixed_center_production_ready",
                "vignette_parameter_contract": "fixed_center",
                "auto_frame_crop_stage_implemented": True,
                "auto_frame_production_ready": False,
                "auto_frame_blocker_code": "vulkan_auto_frame_yunet_unavailable",
                "auto_frame_cpu_tail": True,
                "auto_frame_degraded_reason_code": "vulkan_effect_cpu_tail",
                "auto_frame_selectable_cpu_fallback": False,
                "matting_production_ready": False,
                "matting_reason_code": "open_vulkan_matting_unavailable",
                "matting_blocker_code": "open_vulkan_matting_adapter_unavailable",
                "eye_contact_production_ready": False,
                "eye_contact_reason_code": "open_vulkan_eye_contact_unavailable",
                "eye_contact_blocker_code":
                    "open_vulkan_eye_contact_runtime_unavailable",
                "video_noise_removal_production_ready": False,
                "video_noise_removal_reason_code":
                    "open_vulkan_video_noise_removal_unavailable",
                "video_noise_removal_blocker_code":
                    "open_vulkan_video_noise_removal_runtime_unavailable",
            }}}
        ctl = current_bin / "studiocastctl"
        ctl.write_text(
            "#!/bin/sh\n"
            + "printf '%s\\n' \"$*\" >> " + repr(str(ctl_calls)) + "\n"
            + "printf '%s\\n' " + repr(json.dumps(diagnostics)) + "\n")
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
        self.assertEqual(ctl_calls.read_text().splitlines(),
                         ["diagnostics refresh", "status"])
        self.assertTrue(facts["gpus"]["vulkan"]["compute_device_usable"])
        self.assertEqual(facts["gpus"]["vulkan"]["reason_codes"], ["vulkan_compute_device_usable"])
        self.assertTrue(facts["gpus"]["nvidia"]["cuda_usable"])
        self.assertEqual(facts["runtime"]["onnxruntime"]["probe"], "daemon_runtime_diagnostics")
        self.assertTrue(facts["runtime"]["onnxruntime"]["providers"]["cuda"])
        self.assertEqual(facts["effects"]["capabilities"]["auto_frame"]["maxine"], "production_usable")
        self.assertEqual(facts["effects"]["capabilities"]["auto_frame"]["vulkan"],
                         "usable_with_degraded_behavior")
        self.assertEqual(facts["effects"]["capabilities"]["mirror"]["vulkan"],
                         "production_usable")
        self.assertTrue(facts["effects"]["capabilities"]["mirror"]
                        ["vulkan_evidence"]["production_ready"])
        self.assertEqual(facts["effects"]["capabilities"]["vignette"]
                         ["vulkan_evidence"]["parameter_contract"], "fixed_center")
        self.assertFalse(facts["effects"]["capabilities"]["auto_frame"]
                         ["vulkan_evidence"]["production_ready"])
        self.assertIn("open_vulkan_matting_adapter_unavailable",
                      facts["effects"]["capabilities"]
                      ["virtual_background.blur"]["blocker_reason_codes"]
                      ["vulkan"])

        # An installed older CLI may still return production-looking cached
        # status, but without a successful explicit refresh that evidence is
        # stale/unknown and must not promote an effect.
        current_bin = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/current/bin"
        current_bin.mkdir(parents=True, exist_ok=True)
        ctl_calls = self.s.root / "older-studiocastctl-calls"
        production_looking = {"engines": {"open_vulkan": {
            "compiled_enabled": True, "ok": True,
            "runtime_library_found": True, "instance_created": True,
            "physical_device_found": True, "non_cpu_device_selected": True,
            "compute_queue_available": True, "logical_device_created": True,
            "context_created": True, "context_healthy": True,
            "production_hardware_ready": True, "shader_pipeline_created": True,
            "available_effects": ["mirror"],
            "mirror_production_ready": True,
            "mirror_readiness_code": "open_vulkan_mirror_production_ready",
        }}}
        ctl = current_bin / "studiocastctl"
        ctl.write_text(
            "#!/bin/sh\n"
            + "printf '%s\\n' \"$*\" >> " + repr(str(ctl_calls)) + "\n"
            + "if [ \"${1:-}\" = diagnostics ]; then exit 2; fi\n"
            + "printf '%s\\n' " + repr(json.dumps(production_looking)) + "\n")
        ctl.chmod(0o755)

        facts = self.s.json("analyze", "--target-version", "0.2.9")
        self.assertEqual(ctl_calls.read_text().splitlines(),
                         ["diagnostics refresh", "status"])
        mirror = facts["effects"]["capabilities"]["mirror"]
        self.assertEqual(mirror["vulkan"], "unavailable")
        self.assertFalse(mirror["vulkan_evidence"]["runtime_diagnostics_available"])
        self.assertIn("open_vulkan_runtime_diagnostics_unavailable",
                      mirror["blocker_reason_codes"]["vulkan"])

        ctl_calls = self.s.root / "empty-refresh-studiocastctl-calls"
        ctl.write_text(
            "#!/bin/sh\n"
            + "printf '%s\\n' \"$*\" >> " + repr(str(ctl_calls)) + "\n"
            + "if [ \"${1:-}\" = diagnostics ]; then "
              "printf '%s\\n' '{\"engines\":{}}'; exit 0; fi\n"
            + "printf '%s\\n' " + repr(json.dumps(production_looking)) + "\n")
        ctl.chmod(0o755)
        facts = self.s.json("analyze", "--target-version", "0.2.9")
        self.assertEqual(ctl_calls.read_text().splitlines(),
                         ["diagnostics refresh", "status"])
        mirror = facts["effects"]["capabilities"]["mirror"]
        self.assertEqual(mirror["vulkan"], "unavailable")
        self.assertIn("open_vulkan_runtime_diagnostics_unavailable",
                      mirror["blocker_reason_codes"]["vulkan"])

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

    def test_advanced_flags_are_digest_bound_and_paths_and_ids_fail_closed(self):
        model_root = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/models/alternate"
        plan = self.s.json("plan", "advanced", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--no-service",
            "--model-id", "fastenhancer_s_vd_v1", "--model-destination", str(model_root),
            "--no-open-backends", "--open-audio", "--open-cuda",
            "--v4l-device-number", "22", "--v4l-label", "StudioCast Test Camera",
            "--no-v4l-exclusive-caps", "--allow-unsupported")
        self.assertEqual(plan["route"], "advanced")
        self.assertEqual(plan["desired_state"]["features"]["open_cuda"], True)
        self.assertEqual(plan["desired_state"]["features"]["open_audio"], True)
        self.assertEqual(plan["desired_state"]["model_pack_ids"], ["fastenhancer_s_vd_v1"])
        self.assertEqual(plan["desired_state"]["model_destination"], str(model_root))
        self.assertEqual(plan["desired_state"]["virtual_camera"]["device_number"], 22)
        self.assertFalse(plan["desired_state"]["virtual_camera"]["exclusive_caps"])
        transaction = next(op for op in plan["operations"] if op["kind"] == "model_transaction")
        self.assertEqual(transaction["inputs"]["destination_root"], str(model_root))
        outside = self.s.run("plan", "advanced", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--no-service",
            "--model-id", "fastenhancer_s_vd_v1", "--model-destination", str(self.s.root / "outside"),
            "--allow-unsupported", check=False)
        self.assertEqual(outside.returncode, 2)
        self.assertIn("path.outside_allowed_root", outside.stderr)
        unknown = self.s.run("plan", "advanced", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--no-service",
            "--model-id", "not-a-catalog-pack", "--allow-unsupported", check=False)
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("model.catalog_missing_default", unknown.stderr)

    def test_service_unavailable_degrades_after_core_commit(self):
        facts = base_facts()
        facts["systemd_user"] = {"systemctl_present": False, "manager_usable": False,
                                 "reason_codes": ["service.user_manager.unavailable"]}
        self.s.write_facts(facts)
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--service", "--no-models",
            "--allow-unsupported")
        self.assertIn("service.user_manager.unavailable_degraded",
                      [warning["code"] for warning in plan["warnings"]])
        path = self.s.root / "service-plan.json"; path.write_text(json.dumps(plan))
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.s.facts), check=False,
            env=self._fake_cmake())
        value = terminal_result(result.stdout)
        self.assertEqual((result.returncode, value["state"], value["core_committed"]), (3, "degraded", True))
        self.assertIn("service.state.reconcile", value["retryable_operation_ids"])

    def test_service_unit_uses_xdg_payload_and_preserves_user_edit(self):
        adapter = self.s.root / "fake-systemctl"
        adapter.write_text("#!/bin/sh\nexit 0\n"); adapter.chmod(0o755)
        plan = self.s.json("plan", "install", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--service", "--no-models",
            "--allow-unsupported")
        path = self.s.root / "service-owned.json"; path.write_text(json.dumps(plan))
        env = self._fake_cmake() | {"STUDIOCAST_INSTALLER_TEST_MODE": "1",
                                    "STUDIOCAST_INSTALLER_TEST_SYSTEMCTL": str(adapter)}
        # Restore the sandbox roots overwritten by the fake-tool helper.
        env.update({key: self.s.env[key] for key in
                    ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME")})
        installed = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.s.facts), env=env)
        self.assertEqual(terminal_result(installed.stdout)["state"], "committed")
        unit = Path(self.s.env["XDG_CONFIG_HOME"]) / "systemd/user/studiocastd.service"
        content = unit.read_text()
        self.assertIn(str(Path(self.s.env["XDG_DATA_HOME"]) /
                          "studiocast/current/bin/studiocastd"), content)
        self.assertNotIn("%h/.local/share", content)
        edited = content + "# user edit\n"
        unit.write_text(edited)

        healthy = base_facts(classification="healthy", active_version="0.2.9",
            target_version="0.2.9", version_relation="same",
            desired_configuration={"build_type": "Release",
                "features": {"open_cuda": True, "open_audio": True, "open_vulkan": False},
                "service": {"desired": "enabled_started"},
                "v4l": {"desired": False, "required_for_success": False},
                "model_pack_ids": [], "preserve_settings": True})
        healthy["service"] = {"actual": "active"}
        self.s.write_facts(healthy)
        uninstall = self.s.json("plan", "uninstall", "--facts", str(self.s.facts),
                                "--preserve-user-data")
        path.write_text(json.dumps(uninstall))
        removed = self.s.run("apply-plan", "--plan", str(path), "--digest", uninstall["plan_digest"],
            "--token", uninstall["approval_token"], "--facts", str(self.s.facts), env=env)
        self.assertEqual(terminal_result(removed.stdout)["state"], "committed")
        self.assertEqual(unit.read_text(), edited)

    def test_prefix_spoof_link_is_rejected_and_unexpected_oserror_is_journaled(self):
        spoof = Path(self.s.env["HOME"]) / ".local/bin/studiocast"
        spoof.parent.mkdir(parents=True)
        spoof.symlink_to(Path(self.s.env["XDG_DATA_HOME"]) / "studiocast/current-evil/bin/studiocast")
        plan = self.plan(); path = self.s.root / "spoof.json"; path.write_text(json.dumps(plan))
        rejected = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.s.facts), "--dry-run", check=False)
        self.assertEqual(terminal_result(rejected.stdout)["error"]["code"], "ownership.mismatch")
        spoof.unlink()
        data_root = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"
        if data_root.exists():
            import shutil
            shutil.rmtree(data_root)

        plan = self.plan(); path.write_text(json.dumps(plan))
        only_python = self.s.root / "only-python"; only_python.mkdir()
        (only_python / "python3").symlink_to(sys.executable)
        failed = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
            "--token", plan["approval_token"], "--facts", str(self.s.facts), check=False,
            env=self.s.env | {"PATH": str(only_python)})
        value = terminal_result(failed.stdout)
        self.assertEqual(value["error"]["code"], "operation.unexpected_failure")
        manifest = json.loads((Path(self.s.env["XDG_DATA_HOME"]) /
                               "studiocast/install-manifest.json").read_text())
        entry = next(item for item in manifest["journal"] if item["operation_id"] == "build.configure")
        self.assertEqual((entry["state"], entry["error"]["code"]),
                         ("failed", "operation.unexpected_failure"))

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

    def _fake_cmake(self, fail=False, version="0.2.9"):
        bindir = self.s.root / "fake-bin"; bindir.mkdir(exist_ok=True)
        cmake = bindir / "cmake"
        cmake.write_text("#!/bin/sh\n" + ("exit 41\n" if fail else f"if [ \"$1\" = --build ]; then\n d=$2\n mkdir -p \"$d\"\n for n in studiocast studiocastd studiocastctl studiocast-probe studiocast-open studiocast-maxine studiocast-video studiocast-audio; do printf '#!/bin/sh\\n[ \"$1\" = --version ] && echo StudioCast-{version}\\nexit 0\\n' >\"$d/$n\"; chmod 755 \"$d/$n\"; done\nfi\nexit 0\n"), encoding="utf-8")
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
        facts = self.s.json("analyze", "--target-version", "0.2.9")
        facts["paths"]["free_bytes"] = 20_000_000_000
        self.s.write_facts(facts)
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
        self.assertEqual(terminal_result(result.stdout)["state"], "committed")
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

    def test_post_activation_failure_atomically_rolls_back_and_retains_target(self):
        initial_facts = base_facts(target_version="0.2.8")
        self.s.write_facts(initial_facts)
        initial = self.s.json("plan", "install", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--no-service",
            "--no-models", "--allow-unsupported", "--target-version", "0.2.8")
        path = self.s.root / "rollback-plan.json"
        path.write_text(json.dumps(initial), encoding="utf-8")
        installed = self.s.run("apply-plan", "--plan", str(path),
            "--digest", initial["plan_digest"], "--token", initial["approval_token"],
            "--facts", str(self.s.facts), env=self._fake_cmake(version="0.2.8"))
        self.assertEqual(terminal_result(installed.stdout)["state"], "committed")
        data = Path(self.s.env["XDG_DATA_HOME"]) / "studiocast"
        prior = data / "payloads/0.2.8"
        current = data / "current"
        self.assertEqual(current.resolve(), prior)

        facts = self.s.json("analyze", "--target-version", "0.2.9")
        self.s.write_facts(facts)
        unit = Path(self.s.env["XDG_CONFIG_HOME"]) / "systemd/user/studiocastd.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text("user-owned conflict\n", encoding="utf-8")
        update = self.s.json("plan", "update", "--facts", str(self.s.facts),
            "--source-dir", str(SOURCE), "--no-v4l2loopback", "--service",
            "--no-models", "--allow-unsupported", "--target-version", "0.2.9")
        path.write_text(json.dumps(update), encoding="utf-8")
        failed = self.s.run("apply-plan", "--plan", str(path),
            "--digest", update["plan_digest"], "--token", update["approval_token"],
            "--facts", str(self.s.facts), check=False,
            env=self._fake_cmake(version="0.2.9"))
        value = terminal_result(failed.stdout)
        self.assertEqual(failed.returncode, 2)
        self.assertEqual((value["state"], value["rollback_state"], value["active_version"]),
                         ("rolled_back", "rolled_back", "0.2.8"))
        self.assertEqual(value["error"]["code"], "ownership.mismatch")
        self.assertEqual(current.resolve(), prior)
        self.assertTrue((data / "payloads/0.2.9/bin/studiocast").is_file())
        manifest = json.loads((data / "install-manifest.json").read_text())
        self.assertEqual(manifest["transaction"]["state"], "rolled_back")
        self.assertEqual(manifest["versions"]["active"], "0.2.8")
        rollback_states = [item["state"] for item in manifest["journal"]
                           if item["operation_id"] == "payload.rollback"]
        self.assertEqual(rollback_states, ["rolling_back", "rolled_back"])

    def test_payload_validation_runs_version_probe_before_activation(self):
        plan = self.plan()
        path = self.s.root / "version-plan.json"; path.write_text(json.dumps(plan), encoding="utf-8")
        env = self._fake_cmake()
        cmake = self.s.root / "fake-bin/cmake"
        cmake.write_text(cmake.read_text().replace("StudioCast-0.2.9", "StudioCast-9.9.9"), encoding="utf-8")
        result = self.s.run("apply-plan", "--plan", str(path), "--digest", plan["plan_digest"],
                            "--token", plan["approval_token"], check=False, env=env)
        self.assertEqual(result.returncode, 2)
        value = terminal_result(result.stdout)
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
  # Reap the group-signalled child here; container PID 1 may not reap orphans.
  reap_cancelled_child() {{
    wait "$child"
    exit 130
  }}
  trap reap_cancelled_child INT TERM
  sleep 60 &
  child=$!
  printf '%s' "$child" >'{child_pid}'
  wait "$child"
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
        self.assertEqual(terminal_result(stdout)["state"], "cancelled")
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
            return result, terminal_result(result.stdout) if result.stdout.strip().startswith("{") else None

        success_script = f"""cat >'{captured}'
python3 -c 'import json; p={json.dumps(str(captured))}; r=json.load(open(p)); o=r[\"operations\"][0]; print(json.dumps({{\"schema_version\":1,\"request_id\":r[\"request_id\"],\"transaction_id\":r[\"transaction_id\"],\"plan_digest\":r[\"plan_digest\"],\"state\":\"succeeded\",\"results\":[{{\"id\":o[\"id\"],\"type\":o[\"type\"],\"status\":\"succeeded\",\"files\":[]}}]}}))'
"""
        success, value = invoke(success_script)
        self.assertEqual(success.returncode, 0); self.assertEqual(value["state"], "committed")
        request = json.loads(captured.read_text())
        self.assertEqual(request["policy_version"], 1)
        self.assertEqual(request["preconditions"],
                         {"base_os": "ubuntu", "base_release": "24.04",
                          "kernel_release": "fixture"})
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
        self.assertEqual(terminal_result(result.stdout)["state"], "committed")
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
