"""Pure route and recommendation policy."""

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
    POLICY_VERSION,
    RECOMMENDATION_VERSION,
)

from .packaged import model_download_bytes_from_facts

def route_for(facts: dict[str, Any]) -> tuple[str, str]:
    if not facts.get("os", {}).get("supported", False):
        return "unsupported", "manual_instructions"
    classification = facts.get("installation", {}).get("classification", "absent")
    connectivity = facts.get("connectivity", {}).get("release_source", {}).get("state")
    cache = facts.get("cache", {}).get("release_artifacts", {})
    if connectivity == "offline" and not any(v.get("verified") for v in cache.values() if isinstance(v, dict)):
        return "offline", "stop"
    relation = facts.get("installation", {}).get("version_relation")
    if classification == "absent":
        return "recommended", "install"
    if classification == "healthy":
        if relation == "upgrade":
            return "recommended", "update"
        if relation == "installed_newer":
            return "current_newer", "keep"
        return "modify", "modify"
    if classification in {"partial", "unhealthy", "stale", "unknown_schema"}:
        return "repair", "repair"
    if classification == "tombstone":
        return "repair", "reconstruct"
    return "repair", "repair"


def capability_usable(value: Any) -> bool:
    return value is True or value in {"usable", "production_usable"}


def capability_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(code) for code in value if isinstance(code, str) and code]


def missing_capability_evidence_reason(effect: str, engine: str) -> str:
    return f"effect.{effect}.{engine}.capability_evidence_missing"


def inconsistent_vulkan_evidence_reason(effect: str) -> str:
    return f"effect.{effect}.vulkan.production_evidence_inconsistent"


def vulkan_production_evidence_claimed(matrix: dict[str, Any]) -> bool:
    evidence = matrix.get("vulkan_evidence")
    successes = matrix.get("success_reason_codes")
    return bool(
        capability_usable(matrix.get("vulkan")) or
        (isinstance(evidence, dict) and any(
            evidence.get(key) is True for key in (
                "daemon_production_ready", "effect_production_ready_attested",
                "production_ready"))) or
        (isinstance(successes, dict) and
         capability_reason_codes(successes.get("vulkan"))))


def engine_capability_usable(effect: str, matrix: dict[str, Any], engine: str) -> bool:
    if not capability_usable(matrix.get(engine)):
        return False
    if engine == "cpu":
        cpu_evidence = matrix.get("cpu_evidence")
        return (isinstance(cpu_evidence, dict) and
                cpu_evidence.get("selectable_production_path") is True)
    if engine != "vulkan":
        return True
    evidence = matrix.get("vulkan_evidence")
    if not isinstance(evidence, dict):
        return False
    common_keys = (
        "runtime_diagnostics_available", "backend_compiled",
        "runtime_library_found", "instance_created", "physical_device_found",
        "non_cpu_device_selected", "compute_queue_available",
        "logical_device_created", "context_created", "context_healthy",
        "production_hardware_ready", "utility_kernels_ready", "common_ready",
        "live_stage_implemented", "daemon_production_ready",
        "effect_production_ready_attested",
        "production_ready",
    )
    if not all(evidence.get(key) is True for key in common_keys):
        return False
    required_effect_keys = {
        "mirror": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_layers", "no_dynamic_allocations",
            "no_cpu_readback", "no_cpu_fallback",
        ),
        "vignette": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_layers", "no_dynamic_allocations",
            "no_cpu_readback", "no_cpu_fallback",
        ),
        "virtual_background.blur": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_layers", "no_dynamic_allocations",
            "no_cpu_readback", "no_cpu_fallback",
        ),
        "virtual_background.remove": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_layers", "no_dynamic_allocations",
            "no_cpu_readback", "no_cpu_fallback",
        ),
        "virtual_background.replace": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_layers", "no_dynamic_allocations",
            "no_cpu_readback", "no_cpu_fallback",
        ),
        "virtual_key_light": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_layers", "no_dynamic_allocations",
            "no_cpu_readback", "no_cpu_fallback",
        ),
        "eye_contact": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_readback", "no_cpu_fallback",
        ),
        "video_noise_removal": (
            "runtime_ready", "model_artifact_ready", "warmup_complete",
            "exact_device_and_queue", "device_resident",
            "synchronization_contract_validated", "bounded_reusable_allocations",
            "parity_validated", "no_cpu_readback", "no_cpu_fallback",
            "temporal_history_device_resident", "temporal_history_bounded",
            "history_reset_on_disable", "history_reset_on_reconfigure",
            "capture_sequence_discontinuity_reset",
        ),
    }.get(effect)
    if (required_effect_keys is None or not all(
            evidence.get(key) is True for key in required_effect_keys)):
        return False
    if effect == "vignette" and evidence.get("parameter_contract") != "fixed_center":
        return False
    expected_success = {
        "mirror": "open_vulkan_mirror_production_ready",
        "vignette": "open_vulkan_vignette_fixed_center_production_ready",
        "virtual_background.blur": "open_vulkan_matting_production_ready",
        "virtual_background.remove": "open_vulkan_matting_production_ready",
        "virtual_background.replace": "open_vulkan_matting_production_ready",
        "virtual_key_light": "open_vulkan_matting_production_ready",
        "eye_contact": "open_vulkan_eye_contact_production_ready",
        "video_noise_removal": "open_vulkan_video_noise_removal_production_ready",
    }.get(effect)
    successes = matrix.get("success_reason_codes", {})
    return (expected_success is not None and isinstance(successes, dict) and
            expected_success in successes.get("vulkan", []))


def recommend(facts: dict[str, Any], intent: str, prior: dict[str, Any] | None,
              policy_version: str = POLICY_VERSION, route: str = "recommended") -> dict[str, Any]:
    if facts.get("schema_version") != 1 or facts.get("facts_version") != FACTS_VERSION:
        raise BackendError("facts.unsupported_schema", "Recommendation requires facts v1")
    if policy_version != POLICY_VERSION:
        raise BackendError("policy.unsupported", f"Unsupported policy: {policy_version}")
    detected_route, primary = route_for(facts)
    prior = copy.deepcopy(prior or facts.get("installation", {}).get("desired_configuration") or {})
    prior_service = copy.deepcopy(prior.get("service", {
        "desired": "enabled_started" if facts.get("systemd_user", {}).get("manager_usable") else "disabled"}))
    if not isinstance(prior_service, dict):
        prior_service = {"desired": "unknown"}
    prior_camera = copy.deepcopy(prior.get("v4l", {
        "desired": True, "required_for_success": route == "recommended", "device_number": 10,
        "label": "StudioCast Camera", "exclusive_caps": True, "load": True, "persist": True}))
    if not isinstance(prior_camera, dict):
        prior_camera = {"desired": "unknown"}
    prior_camera.setdefault("load", True)
    prior_camera.setdefault("persist", True)
    selections: dict[str, Any] = {
        "scope": "user", "source": {"kind": "official_release", "verified_required": True},
        "build_type": prior.get("build_type", "Release"),
        "service": prior_service, "virtual_camera": prior_camera,
        "model_pack_ids": list(prior.get("model_pack_ids", DEFAULT_MODEL_PACKS)),
        "model_destination": prior.get("model_destination"),
        "model_source": prior.get("model_source"),
        "model_offline": bool(prior.get("model_offline", False)),
        "effects": {}, "features": copy.deepcopy(prior.get("features", {})),
        "shader_tools": False, "mesa_packages": False,
        "manage_dependencies": bool(prior.get("manage_dependencies", route == "recommended")),
        "preserve_settings": prior.get("preserve_settings", True),
    }
    if route == "recommended":
        selections["model_pack_ids"] = list(DEFAULT_MODEL_PACKS)
        if type(selections["virtual_camera"].get("desired")) is bool:
            selections["virtual_camera"].update({"desired": True, "required_for_success": True})
    raw_capabilities = facts.get("effects", {}).get("capabilities", {})
    capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    ordered_effects = list(CANONICAL_EFFECTS)
    ordered_effects.extend(sorted(
        str(effect) for effect in capabilities if effect not in CANONICAL_EFFECTS))
    selected_vulkan = False
    for effect in ordered_effects:
        raw_matrix = capabilities.get(effect)
        matrix = raw_matrix if isinstance(raw_matrix, dict) else {}
        selected = "unavailable"
        for engine in ("maxine", "cuda", "vulkan", "cpu"):
            if engine_capability_usable(effect, matrix, engine):
                selected = engine
                break
        raw_successes = matrix.get("success_reason_codes", {})
        raw_blockers = matrix.get("blocker_reason_codes", {})
        raw_successes = raw_successes if isinstance(raw_successes, dict) else {}
        raw_blockers = raw_blockers if isinstance(raw_blockers, dict) else {}
        successes = {
            engine: capability_reason_codes(raw_successes.get(engine))
            for engine in ("maxine", "cuda", "vulkan", "cpu")
        }
        blockers_by_engine = {
            engine: capability_reason_codes(raw_blockers.get(engine))
            for engine in ("maxine", "cuda", "vulkan", "cpu")
        }
        for engine in ("maxine", "cuda", "vulkan", "cpu"):
            if (not engine_capability_usable(effect, matrix, engine) and
                    not blockers_by_engine[engine]):
                if engine == "vulkan" and vulkan_production_evidence_claimed(matrix):
                    blockers_by_engine[engine].append(
                        inconsistent_vulkan_evidence_reason(effect))
                else:
                    blockers_by_engine[engine].append(
                        missing_capability_evidence_reason(effect, engine))
        selected_codes = list(successes.get(selected, [])) if selected != "unavailable" else [
            code for engine in ("maxine", "cuda", "vulkan", "cpu")
            for code in blockers_by_engine.get(engine, [])]
        selections["effects"][effect] = {
            "engine_precedence": ["maxine", "cuda", "vulkan", "cpu"],
            "selected": selected,
            "selected_reason_codes": selected_codes,
            "capability_success_reason_codes": copy.deepcopy(successes),
            "capability_blocker_reason_codes": copy.deepcopy(blockers_by_engine),
        }
        selected_vulkan = selected_vulkan or selected == "vulkan"
    if route == "recommended" and detected_route == "recommended":
        selections["features"]["open_vulkan"] = selected_vulkan
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not facts.get("os", {}).get("supported", False):
        blockers.append({"code": "os.unsupported.automatic_changes_blocked", "applies_to": "installation"})
    free = facts.get("paths", {}).get("free_bytes", 0)
    base_required = sum(facts.get("paths", {}).get("required_bytes", {}).values())
    declared_model_bytes = model_download_bytes_from_facts(facts, selections["model_pack_ids"])
    model_download_bytes = declared_model_bytes if declared_model_bytes is not None else 0
    if declared_model_bytes is None and route == "recommended" and selections["model_pack_ids"]:
        blockers.append({"code": "artifact_sizes_untrusted", "applies_to": "models"})
    required = base_required + model_download_bytes
    if free < required:
        blockers.append({"code": "disk.staging.insufficient", "applies_to": "paths"})
    if detected_route == "offline" and route == "recommended":
        blockers.append({"code": "network.release.offline_no_verified_cache", "applies_to": "source"})
    if selections["service"].get("desired") not in {"enabled_started", "disabled"}:
        blockers.append({"code": "configuration.service.selection_required", "applies_to": "service_changes"})
    camera = selections["virtual_camera"]
    if type(camera.get("desired")) is not bool:
        blockers.append({"code": "configuration.virtual_camera.selection_required",
                         "applies_to": "virtual_camera"})
    elif camera.get("desired"):
        v4l = facts.get("v4l2", {})
        if v4l.get("device_number_conflict"):
            blockers.append({"code": "v4l.number.conflict", "applies_to": "virtual_camera"})
        if (not v4l.get("kernel_headers_usable", True) or not v4l.get("dkms_viable", True)) and not v4l.get("module_available"):
            blockers.append({"code": "v4l.dkms.not_viable", "applies_to": "virtual_camera"})
        needs_change = not (v4l.get("device_present") and v4l.get("module_loaded") and (not camera.get("persist") or v4l.get("persistence_state") == "configured"))
        helper = facts.get("privileged_helper", {})
        if needs_change and not (helper.get("trusted", False) and
                                 helper.get("authorization_transport_available", helper.get("trusted", False))):
            blockers.append({"code": helper.get("reason_codes", ["privileged_helper_missing"])[0],
                             "applies_to": "virtual_camera"})
    else:
        warnings.append({"code": "install.degraded.no_virtual_camera", "applies_to": "virtual_camera"})
    reasons = [
        {"code": "scope.user_local.required", "applies_to": "scope"},
        {"code": "source.official.required", "applies_to": "source"},
        {"code": "models.open_fallbacks.default", "applies_to": "models"},
        {"code": "effects.precedence.per_effect", "applies_to": "effects"},
    ]
    return {"schema_version": 1, "recommendation_version": RECOMMENDATION_VERSION,
        "policy_version": policy_version, "facts_fingerprint": facts["host_fingerprint"],
        "intent": intent, "route": detected_route if route == "recommended" else route,
        "selection_route": route,
        "primary_action": primary, "selections": selections,
        "disk": {"free_bytes": free, "base_required_bytes": base_required,
                 "download_bytes": model_download_bytes, "required_bytes": required},
        "reasons": reasons, "blockers": blockers, "warnings": warnings,
        "alternatives": [{"code": "route.custom", "action": "customize"}]}
