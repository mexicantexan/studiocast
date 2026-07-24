"""Fail-closed parser for the additive typed multi-island video status.

The installer consumes this snapshot only as shadow evidence.  Existing
per-effect recommendation facts remain authoritative until a separately
reviewed migration proves parity.  Missing or incomplete prepared-route
authority is rejected rather than interpreted as compatibility-route state.
"""

from __future__ import annotations

from typing import Any

from .contracts import CANONICAL_EFFECTS


SCHEMA_VERSION = 1
HAL_CONTRACT_VERSION = 1
REASON_CODE_VERSION = 1
MAX_ISLANDS = 8
MAX_STAGES = 32
MAX_CAPABILITY_EVIDENCE = 24
MAX_UINT64 = (1 << 64) - 1
MAX_UINT32 = (1 << 32) - 1

EFFECT_IDS = {
    "video_noise_removal": 1,
    "eye_contact": 2,
    "virtual_background.blur": 3,
    "virtual_background.remove": 4,
    "virtual_background.replace": 5,
    "virtual_key_light": 6,
    "auto_frame": 7,
    "vignette": 8,
    "mirror": 9,
}
EFFECT_IDENTITIES = {0: "", **{identifier: token for token, identifier in EFFECT_IDS.items()}}
BACKENDS = {
    0: "", 1: "cpu", 2: "cuda", 3: "maxine",
    4: "open_video_onnxruntime", 5: "vulkan",
}
PROVIDERS = {
    0: "", 1: "maxine_vfx", 2: "maxine_ar",
    3: "open_video_onnxruntime_cuda",
    4: "open_video_onnxruntime_cpu",
    5: "open_vulkan_production_kernels",
    6: "maxine_cuda_utility",
}
STATUSES = {
    0: "invalid", 1: "ready", 2: "blocked", 3: "degraded",
    4: "unavailable", 5: "passthrough", 6: "executed", 7: "failed",
    8: "quarantined", 9: "stopped", 10: "resource_exhausted", 11: "stale",
}
REASONS = {
    0: "", 1: "malformed_contract", 2: "unsupported_contract_version",
    3: "malformed_descriptor", 4: "malformed_execution_domain",
    5: "foreign_adapter", 6: "stale_program", 7: "wrong_generation",
    8: "wrong_execution_domain", 9: "resource_exhausted",
    10: "cross_adapter_residency_forbidden", 11: "required_stage_failed",
    12: "optional_stage_failed_open", 13: "device_lost",
    14: "temporal_discontinuity", 15: "reconfiguration_required",
    16: "shutdown_requested", 17: "output_write_failed",
    18: "output_not_committed", 19: "incomplete_execution",
    20: "invalid_resource", 21: "synchronization_failed",
    22: "strict_backend_unavailable", 23: "cpu_tail_required",
    24: "circuit_breaker_open", 25: "cancellation_unsupported_v1",
    26: "semantic_provider_unavailable", 27: "semantic_effect_unsupported",
    28: "capability_evidence_incomplete",
    29: "host_frame_binding_unsupported", 30: "host_frame_binding_invalid",
    31: "circuit_breaker_retry_pending",
    32: "circuit_breaker_permanently_disabled",
    33: "old_program_policy_incompatible",
    34: "transfer_contract_violated", 35: "frame_order_violated",
}
ROUTES = {
    1: "final_visual",
    2: "maxine_resident",
    3: "open_cuda_virtual_background",
}
REQUEST_MODES = {1: "automatic", 2: "explicit_backend"}
LIFECYCLES = {
    0: "disabled", 1: "idle", 2: "starting",
    3: "running", 4: "stopping", 5: "failed",
}
STAGE_STATUSES = {
    0: "not_run", 1: "succeeded", 2: "optional_failed_open",
    3: "required_failed", 4: "disabled_preserved",
}
FAILURE_ACTIONS = {
    1: "fail_closed",
    2: "fail_open_preserve_input",
    3: "fail_open_preserve_last_valid",
}
API_FAMILIES = {
    0: "invalid", 1: "host", 2: "cuda_driver",
    3: "maxine_nvcv", 4: "onnxruntime", 5: "vulkan",
}
ALLOCATOR_FAMILIES = dict(API_FAMILIES)
MEMORY_HANDLE_FAMILIES = {
    0: "invalid", 1: "host_pointer", 2: "cuda_device_pointer",
    3: "nvcv_image", 4: "ort_value", 5: "vulkan_buffer",
    6: "vulkan_image",
}
OWNERSHIP_MODELS = {
    0: "invalid", 1: "adapter_exclusive",
    2: "borrowed_until_completion", 3: "externally_owned",
}
SYNCHRONIZATION_PROTOCOLS = {
    0: "invalid", 1: "host_synchronous", 2: "cuda_stream_ordered",
    3: "maxine_bound_cuda_stream", 4: "ort_provider_synchronous",
    5: "vulkan_queue_fence",
}
LANES = {0: "", 1: "maxine", 2: "open_cuda", 3: "open_vulkan", 4: "cpu"}
DISPOSITIONS = {
    "invalid", "production_usable", "usable_with_degraded_behavior",
    "experimental", "diagnostics_only", "unavailable",
}
CPU_TAILS = {
    0: "", 1: "output_format_conversion", 2: "auto_frame_tracking",
    3: "provider_postprocess", 4: "writer_copy",
}
CPU_TAIL_PLACEMENTS = {1: "after_stage", 2: "before_stage"}
CAPABILITY_EVIDENCE_CODES = {
    0: "", 1: "vulkan_build_enabled", 2: "vulkan_loader_available",
    3: "vulkan_physical_device_available", 4: "vulkan_non_cpu_device",
    5: "vulkan_compute_queue_available", 6: "vulkan_context_healthy",
    7: "vulkan_utility_kernels_ready", 8: "vulkan_exact_device_shared",
    9: "vulkan_device_local_residency", 10: "vulkan_bounded_allocations",
    11: "vulkan_no_cpu_layers", 12: "vulkan_no_intermediate_readbacks",
    13: "open_vulkan_mirror_production_ready",
    14: "open_vulkan_vignette_fixed_center_production_ready",
    15: "vulkan_matting_runtime_ready",
    16: "vulkan_model_artifacts_valid", 17: "vulkan_warmup_complete",
    18: "vulkan_live_stage_present",
    19: "vulkan_tracked_vignette_center_supported",
    20: "provider_runtime_available", 21: "model_artifact_identity_valid",
    22: "synchronization_evidence_valid",
    23: "import_export_interop_valid",
    24: "vulkan_backend_disabled_in_build",
    25: "vulkan_runtime_not_found", 26: "vulkan_loader_unavailable",
    27: "vulkan_no_physical_device", 28: "vulkan_no_compute_queue",
    29: "vulkan_only_cpu_devices_available",
    30: "vulkan_requested_device_not_found",
    31: "vulkan_requested_device_no_compute_queue",
    32: "vulkan_requested_device_ambiguous",
    33: "vulkan_device_create_failed",
    34: "vulkan_context_uninitialized", 35: "vulkan_device_lost",
    36: "vulkan_production_hardware_not_ready",
    37: "open_vulkan_utility_kernels_unavailable",
    38: "open_vulkan_runtime_diagnostics_unavailable",
    39: "open_vulkan_matting_unavailable",
    40: "open_vulkan_matting_adapter_unavailable",
    41: "open_vulkan_eye_contact_runtime_unavailable",
    42: "open_vulkan_video_noise_removal_runtime_unavailable",
    43: "vulkan_auto_frame_yunet_unavailable",
    44: "vulkan_vignette_tracked_center_not_supported",
    45: "vulkan_production_evidence_inconsistent",
    46: "per_effect_capability_evidence_missing",
    47: "open_vulkan_eye_contact_unavailable",
    48: "open_vulkan_video_noise_removal_unavailable",
    49: "vulkan_effect_cpu_tail",
}
CAPABILITY_CODES = {
    0: "", 1: "maxine.effect.usable", 2: "maxine.effect.unavailable",
    3: "open_cuda_effect_available", 4: "open_cuda_effect_unavailable",
    10: "effect.video_noise_removal.cpu.no_selectable_production_path",
    11: "effect.eye_contact.cpu.no_selectable_production_path",
    12: "effect.virtual_background.blur.cpu.no_selectable_production_path",
    13: "effect.virtual_background.remove.cpu.no_selectable_production_path",
    14: "effect.virtual_background.replace.cpu.no_selectable_production_path",
    15: "effect.virtual_key_light.cpu.no_selectable_production_path",
    16: "effect.auto_frame.cpu.no_selectable_production_path",
    17: "effect.vignette.cpu.no_selectable_production_path",
    18: "effect.mirror.cpu.no_selectable_production_path",
    100: "open_vulkan_mirror_production_ready",
    101: "open_vulkan_vignette_fixed_center_production_ready",
    102: "vulkan_backend_disabled_in_build",
    103: "vulkan_runtime_not_found", 104: "vulkan_loader_unavailable",
    105: "vulkan_no_physical_device", 106: "vulkan_no_compute_queue",
    107: "vulkan_only_cpu_devices_available",
    108: "vulkan_requested_device_not_found",
    109: "vulkan_requested_device_no_compute_queue",
    110: "vulkan_requested_device_ambiguous",
    111: "vulkan_device_create_failed",
    112: "vulkan_context_uninitialized", 113: "vulkan_device_lost",
    114: "vulkan_production_hardware_not_ready",
    115: "open_vulkan_utility_kernels_unavailable",
    116: "open_vulkan_runtime_diagnostics_unavailable",
    117: "open_vulkan_matting_unavailable",
    118: "open_vulkan_matting_adapter_unavailable",
    119: "open_vulkan_eye_contact_unavailable",
    120: "open_vulkan_eye_contact_runtime_unavailable",
    121: "open_vulkan_video_noise_removal_unavailable",
    122: "open_vulkan_video_noise_removal_runtime_unavailable",
    123: "vulkan_auto_frame_yunet_unavailable",
    124: "vulkan_vignette_tracked_center_not_supported",
    125: "vulkan_production_evidence_inconsistent",
    126: "per_effect_capability_evidence_missing",
    127: "vulkan_effect_cpu_tail",
    128: "open_vulkan_mirror_unavailable",
    129: "open_vulkan_vignette_unavailable",
    130: "vulkan_instance_create_failed",
    131: "open_vulkan_effect_evidence_not_defined",
    132: "open_vulkan_runtime_evidence_inconsistent",
    133: "open_vulkan_mirror_production_evidence_missing",
    134: "open_vulkan_mirror_production_evidence_inconsistent",
    135: "open_vulkan_mirror_live_stage_unavailable",
    136: "open_vulkan_mirror_live_stage_not_active",
    137: "open_vulkan_vignette_production_evidence_missing",
    138: "open_vulkan_vignette_production_evidence_inconsistent",
    139: "open_vulkan_vignette_live_stage_unavailable",
    140: "open_vulkan_vignette_live_stage_not_active",
    141: "open_vulkan_matting_production_ready",
    142: "open_vulkan_eye_contact_production_ready",
    143: "open_vulkan_video_noise_removal_production_ready",
    144: "vulkan_probe_failed", 145: "vulkan_probe_tool_unavailable",
    200: "capability_snapshot_current",
    201: "capability_refresh_not_run",
    202: "capability_refresh_in_progress",
    203: "capability_refresh_failed",
    204: "capability_snapshot_stale",
    205: "capability_structure_invalid",
}

TRANSFER_KEYS = (
    "uploads", "final_readbacks", "intermediate_readbacks",
    "intermediate_uploads", "synchronizations", "cpu_tail_boundaries",
)
GENERATION_KEYS = ("capability", "configuration", "provider", "device", "context")
PROGRAM_GENERATION_KEYS = (
    "configuration", "geometry", "model", "provider", "device", "context",
)


class _InvalidStatus(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise _InvalidStatus(code)


def _object(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _array(value: Any, code: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        _fail(code)
    return value


def _uint(value: Any, code: str, *, nonzero: bool = False) -> int:
    if (isinstance(value, bool) or not isinstance(value, int) or value < 0 or
            value > MAX_UINT64):
        _fail(code)
    if nonzero and value == 0:
        _fail(code)
    return value


def _boolean(value: Any, code: str) -> bool:
    if not isinstance(value, bool):
        _fail(code)
    return value


def _pair(value: Any, known: dict[int, str], code: str) -> tuple[int, str]:
    item = _object(value, code)
    identifier = _uint(item.get("id"), code)
    token = item.get("token")
    if not isinstance(token, str) or known.get(identifier) != token:
        _fail(code)
    return identifier, token


def _generations(value: Any, keys: tuple[str, ...], code: str) -> dict[str, int]:
    item = _object(value, code)
    return {key: _uint(item.get(key), code, nonzero=True) for key in keys}


def _commit(value: Any, code: str) -> dict[str, Any]:
    item = _object(value, code)
    valid = _boolean(item.get("valid"), code)
    capture = _uint(item.get("capture_sequence"), code)
    output = _uint(item.get("output_commit_sequence"), code)
    if valid != (capture > 0 and output > 0):
        _fail(code)
    return {
        "valid": valid,
        "capture_sequence": capture,
        "output_commit_sequence": output,
    }


def _transfers(value: Any, code: str) -> dict[str, int]:
    item = _object(value, code)
    return {key: _uint(item.get(key), code) for key in TRANSFER_KEYS}


def _invalidation_generations(value: Any, code: str) -> dict[str, int]:
    item = _object(value, code)
    generations = _generations(item.get("generations"), PROGRAM_GENERATION_KEYS, code)
    for key in ("canonical_plan_hash", "model_artifact_hash", "execution_domain_hash"):
        _uint(item.get(key), code)
    return generations


def _domain(value: Any, code: str) -> dict[str, Any]:
    item = _object(value, code)
    backend_id, backend = _pair(item.get("backend"), BACKENDS, code)
    api_id, api = _pair(item.get("api_family"), API_FAMILIES, code)
    allocator_id, allocator = _pair(
        item.get("allocator_family"), ALLOCATOR_FAMILIES, code)
    memory_id, memory = _pair(
        item.get("memory_handle_family"), MEMORY_HANDLE_FAMILIES, code)
    ownership_id, ownership = _pair(
        item.get("ownership_model"), OWNERSHIP_MODELS, code)
    sync_id, sync = _pair(
        item.get("synchronization_protocol"), SYNCHRONIZATION_PROTOCOLS, code)
    physical = _object(item.get("physical_device"), code)
    vendor_id = _uint(physical.get("vendor_id"), code)
    device_id = _uint(physical.get("device_id"), code)
    if vendor_id > MAX_UINT32 or device_id > MAX_UINT32:
        _fail(code)
    stable_id = physical.get("stable_id")
    if (not isinstance(stable_id, str) or len(stable_id) != 32 or
            any(character not in "0123456789abcdef" for character in stable_id) or
            stable_id == "0" * 32):
        _fail(code)
    context = _uint(item.get("logical_context_generation"), code, nonzero=True)
    queue = _uint(item.get("queue_or_stream_identity"), code, nonzero=True)
    adapter = _uint(item.get("adapter_instance_id"), code, nonzero=True)
    exact_domains = {
        2: (2, 2, {2}, 1, 2),
        3: (3, 3, {3}, 1, 3),
        5: (5, 5, {5, 6}, 1, 5),
    }
    expected = exact_domains.get(backend_id)
    if (expected is None or api_id != expected[0] or
            allocator_id != expected[1] or memory_id not in expected[2] or
            ownership_id != expected[3] or sync_id != expected[4]):
        _fail(code)
    return {
        "backend": backend, "backend_id": backend_id, "api_family": api,
        "allocator_family": allocator, "memory_handle_family": memory,
        "ownership_model": ownership, "synchronization_protocol": sync,
        "physical_device": {
            "vendor_id": vendor_id, "device_id": device_id,
            "stable_id": stable_id,
        },
        "logical_context_generation": context,
        "queue_or_stream_identity": queue,
        "adapter_instance_id": adapter,
    }


def _same_domain(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def _capability_code(value: Any, code: str) -> dict[str, Any]:
    identifier, token = _pair(value, CAPABILITY_CODES, code)
    return {"id": identifier, "token": token}


def _semantic_evidence(
    value: Any,
    code: str,
    *,
    matrix_shape: bool = False,
) -> list[dict[str, Any]]:
    values = _array(value, code, MAX_CAPABILITY_EVIDENCE)
    parsed = []
    seen: set[tuple[int, int]] = set()
    for raw in values:
        item = _object(raw, code)
        if matrix_shape:
            evidence_id = _uint(item.get("code_id"), code)
            evidence_token = item.get("code")
            status_id = _uint(item.get("status_id"), code)
            status_token = item.get("status")
            reason_id = _uint(item.get("reason_id"), code)
            reason_token = item.get("reason")
            effect_token = item.get("effect")
            backend_token = item.get("backend")
            effect_id = EFFECT_IDS.get(effect_token)
            backend_id = next(
                (identifier for identifier, token in BACKENDS.items()
                 if token == backend_token), None)
            if (CAPABILITY_EVIDENCE_CODES.get(evidence_id) != evidence_token or
                    STATUSES.get(status_id) != status_token or
                    REASONS.get(reason_id) != reason_token or
                    effect_id is None or backend_id is None):
                _fail(code)
        else:
            evidence_id, evidence_token = _pair(
                item.get("code"), CAPABILITY_EVIDENCE_CODES, code)
            status_id, status_token = _pair(
                item.get("status"), STATUSES, code)
            reason_id, reason_token = _pair(
                item.get("reason"), REASONS, code)
            effect_id, effect_token = _pair(
                item.get("effect"), EFFECT_IDENTITIES, code)
            backend_id, backend_token = _pair(
                item.get("backend"), BACKENDS, code)
        satisfied = _boolean(item.get("satisfied"), code)
        if evidence_id == 0 or status_id == 0 or backend_id == 0 or effect_id == 0:
            _fail(code)
        if satisfied:
            if not ((status_id == 1 and reason_id == 0) or
                    (status_id == 3 and reason_id != 0)):
                _fail(code)
        elif status_id not in {2, 4} or reason_id == 0:
            _fail(code)
        identity = (evidence_id, effect_id)
        if identity in seen:
            _fail(code)
        seen.add(identity)
        parsed.append({
            "code": {"id": evidence_id, "token": evidence_token},
            "status": {"id": status_id, "token": status_token},
            "reason": {"id": reason_id, "token": reason_token},
            "effect": {"id": effect_id, "token": effect_token},
            "backend": {"id": backend_id, "token": backend_token},
            "satisfied": satisfied,
        })
    return parsed


def _capability_selection(value: Any, effect: str, code: str) -> dict[str, Any]:
    item = _object(value, code)
    lane_id, lane = _pair(item.get("lane"), LANES, code)
    capability = _capability_code(item.get("code"), code)
    nested = _capability_code(item.get("nested_code"), code)
    semantic = _object(item.get("semantic"), code)
    semantic_effect_id, semantic_effect = _pair(
        semantic.get("effect", semantic.get("semantic")), EFFECT_IDENTITIES, code)
    provider_id, provider = _pair(semantic.get("provider"), PROVIDERS, code)
    status_id, status = _pair(semantic.get("status"), STATUSES, code)
    reason_id, reason = _pair(semantic.get("reason"), REASONS, code)
    cpu_tail_id, cpu_tail = _pair(semantic.get("cpu_tail"), CPU_TAILS, code)
    _uint(semantic.get("cpu_tail_scratch_bytes"), code)
    _pair(
        semantic.get("cpu_tail_placement"), CPU_TAIL_PLACEMENTS, code)
    live_stage = _boolean(semantic.get("live_stage"), code)
    production_eligible = _boolean(semantic.get("production_eligible"), code)
    exact_evidence = _boolean(semantic.get("exact_evidence"), code)
    _boolean(semantic.get("supports_tracked_vignette_center"), code)
    evidence = _semantic_evidence(semantic.get("evidence"), code)
    if lane_id == 0:
        if (capability["id"] != 0 or nested["id"] != 0 or
                semantic_effect_id != 0 or provider_id != 0 or status_id != 0 or
                live_stage or production_eligible or exact_evidence or evidence):
            _fail(code)
    elif semantic_effect != effect:
        _fail(code)
    return {
        "lane": lane, "lane_id": lane_id, "code": capability,
        "nested_code": nested, "provider": provider, "provider_id": provider_id,
        "status": status, "status_id": status_id,
        "reason": reason, "reason_id": reason_id,
        "effect_id": semantic_effect_id, "cpu_tail": cpu_tail,
        "cpu_tail_id": cpu_tail_id, "live_stage": live_stage,
        "production_eligible": production_eligible,
        "exact_evidence": exact_evidence, "evidence": evidence,
    }


def _capability_matrix(
    capabilities: dict[str, Any],
    code: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    lanes = _array(
        capabilities.get("lanes"), code, len(EFFECT_IDS) * 4)
    if len(lanes) != len(EFFECT_IDS) * 4:
        _fail(code)
    parsed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in lanes:
        item = _object(raw, code)
        effect = item.get("effect")
        effect_id = item.get("effect_id")
        lane = item.get("lane")
        lane_id = item.get("lane_id")
        if (not isinstance(effect, str) or EFFECT_IDS.get(effect) != effect_id or
                isinstance(lane_id, bool) or not isinstance(lane_id, int) or
                LANES.get(lane_id) != lane or lane_id == 0):
            _fail(code)
        disposition = item.get("disposition")
        if disposition not in DISPOSITIONS:
            _fail(code)
        capability = _capability_code(item.get("code"), code)
        nested = _capability_code(item.get("nested_code"), code)
        selectable = _boolean(item.get("selectable"), code)
        semantic = _object(item.get("semantic"), code)
        provider_id = _uint(semantic.get("provider_id"), code)
        provider = semantic.get("provider")
        status_id = _uint(semantic.get("status_id"), code)
        status = semantic.get("status")
        reason_id = _uint(semantic.get("reason_id"), code)
        reason = semantic.get("reason")
        if (PROVIDERS.get(provider_id) != provider or
                STATUSES.get(status_id) != status or
                REASONS.get(reason_id) != reason):
            _fail(code)
        live_stage = _boolean(semantic.get("live_stage"), code)
        production_eligible = _boolean(
            semantic.get("production_eligible"), code)
        exact_evidence = _boolean(semantic.get("exact_evidence"), code)
        cpu_tail = semantic.get("cpu_tail")
        cpu_tail_id = next(
            (identifier for identifier, token in CPU_TAILS.items()
             if token == cpu_tail), None)
        if cpu_tail_id is None:
            _fail(code)
        _uint(semantic.get("cpu_tail_scratch_bytes"), code)
        _boolean(semantic.get("supports_tracked_vignette_center"), code)
        evidence = _semantic_evidence(
            semantic.get("evidence"), code, matrix_shape=True)
        key = (effect, lane_id)
        if key in parsed:
            _fail(code)
        parsed[key] = {
            "lane": lane, "lane_id": lane_id, "code": capability,
            "nested_code": nested, "provider": provider,
            "provider_id": provider_id, "status": status,
            "status_id": status_id, "reason": reason,
            "reason_id": reason_id, "effect_id": effect_id,
            "cpu_tail": cpu_tail, "cpu_tail_id": cpu_tail_id,
            "live_stage": live_stage,
            "production_eligible": production_eligible,
            "exact_evidence": exact_evidence, "evidence": evidence,
            "selectable": selectable, "disposition": disposition,
        }
    expected = {
        (effect, lane_id)
        for effect in EFFECT_IDS
        for lane_id in (1, 2, 3, 4)
    }
    if set(parsed) != expected:
        _fail(code)
    return parsed


def _capability_matches(
    selection: dict[str, Any],
    lane: dict[str, Any],
) -> bool:
    return all(selection[key] == lane[key] for key in (
        "lane_id", "code", "nested_code", "provider_id", "status_id",
        "reason_id", "effect_id", "cpu_tail_id", "live_stage",
        "production_eligible", "exact_evidence", "evidence",
    ))


def _selected_capability_is_exact(
    effect: str,
    selection: dict[str, Any],
) -> bool:
    if (not selection["live_stage"] or
            not selection["production_eligible"] or
            not selection["exact_evidence"] or
            selection["status_id"] not in {1, 3}):
        return False
    if selection["lane_id"] != 3:
        return True
    if effect not in {"mirror", "vignette"}:
        return False
    required = set(range(1, 13))
    required.add(13 if effect == "mirror" else 14)
    satisfied = {
        item["code"]["id"]
        for item in selection["evidence"]
        if item["satisfied"] and item["status"]["id"] == 1 and
        item["reason"]["id"] == 0 and
        item["effect"]["token"] == effect and
        item["backend"]["id"] == 5
    }
    return required <= satisfied


def _stage_status(value: Any, code: str) -> tuple[int, str]:
    return _pair(value, STAGE_STATUSES, code)


def _parse_island(
    value: Any,
    status_generation: int,
    capability_generation: int,
    program_generations: dict[str, int],
    writer_commit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    code = "typed_status.island_invalid"
    item = _object(value, code)
    route_id, route = _pair(item.get("route"), ROUTES, code)
    island_id = _uint(item.get("island_id"), code, nonzero=True)
    adapter = _uint(item.get("adapter_instance_id"), code, nonzero=True)
    if (_uint(item.get("snapshot_generation"), code, nonzero=True) !=
            status_generation or
            _uint(item.get("capability_generation"), code, nonzero=True) !=
            capability_generation):
        _fail("typed_status.stale_generation")
    backend_id, backend = _pair(item.get("backend"), BACKENDS, code)
    valid = _boolean(item.get("valid"), code)
    current = _boolean(item.get("current"), code)
    if not valid or not current:
        _fail("typed_status.stale_island")
    domain = _domain(item.get("execution_domain"), code)
    if domain["backend_id"] != backend_id or domain["adapter_instance_id"] != adapter:
        _fail("typed_status.domain_mismatch")
    expected_backend = {1: 5, 2: 3, 3: 2}[route_id]
    expected_providers = {1: {5}, 2: {1, 2, 6}, 3: {3}}[route_id]
    if backend_id != expected_backend:
        _fail("typed_status.route_backend_mismatch")

    resolution = _object(item.get("resolution"), code)
    _pair(resolution.get("request_mode"), REQUEST_MODES, code)
    resolution_status_id, _ = _pair(
        resolution.get("status"), STATUSES, code)
    resolution_reason_id, _ = _pair(
        resolution.get("reason"), REASONS, code)
    requested_backend_id, _ = _pair(
        resolution.get("requested_backend"), BACKENDS, code)
    selected_backend_id, _ = _pair(
        resolution.get("selected_backend"), BACKENDS, code)
    resolution_active_backend_id, _ = _pair(
        resolution.get("active_backend"), BACKENDS, code)
    provider_id, provider = _pair(
        resolution.get("selected_provider"), PROVIDERS, code)
    strict_backend = _boolean(resolution.get("strict_backend_selection"), code)
    resolution_degraded = _boolean(resolution.get("degraded"), code)
    _boolean(resolution.get("fallback"), code)
    resolution_passthrough = _boolean(resolution.get("passthrough"), code)
    if (resolution_status_id not in {1, 3} or
            resolution_degraded != (resolution_status_id == 3) or
            (resolution_status_id == 1 and resolution_reason_id != 0) or
            resolution_passthrough or selected_backend_id != backend_id or
            resolution_active_backend_id not in {0, backend_id} or
            provider_id not in expected_providers):
        _fail("typed_status.island_selection_mismatch")
    if strict_backend and requested_backend_id != selected_backend_id:
        _fail("typed_status.strict_backend_substitution")
    if not _boolean(resolution.get("available"), code):
        _fail("typed_status.current_island_unavailable")
    resolution_stages = _array(resolution.get("stages"), code, MAX_STAGES)
    parsed_resolution_stages: dict[int, dict[str, Any]] = {}
    for raw_stage in resolution_stages:
        stage = _object(raw_stage, code)
        stage_id = _uint(stage.get("stage_id"), code, nonzero=True)
        effect_id, effect = _pair(
            stage.get("effect"), EFFECT_IDENTITIES, code)
        required = _boolean(stage.get("required"), code)
        failure_action_id, _ = _pair(
            stage.get("failure_action"), FAILURE_ACTIONS, code)
        _boolean(stage.get("requires_tracked_vignette_center"), code)
        stage_status_id, _ = _pair(stage.get("status"), STATUSES, code)
        stage_reason_id, _ = _pair(stage.get("reason"), REASONS, code)
        stage_backend_id, _ = _pair(
            stage.get("selected_backend"), BACKENDS, code)
        stage_provider_id, _ = _pair(stage.get("provider"), PROVIDERS, code)
        stage_adapter = _uint(
            stage.get("adapter_instance_id"), code, nonzero=True)
        stage_domain = _domain(stage.get("execution_domain"), code)
        _pair(stage.get("cpu_tail"), CPU_TAILS, code)
        _uint(stage.get("cpu_tail_scratch_bytes"), code)
        _pair(
            stage.get("cpu_tail_placement"), CPU_TAIL_PLACEMENTS, code)
        _boolean(stage.get("skipped_optional"), code)
        _boolean(stage.get("fallback_selected"), code)
        _semantic_evidence(stage.get("evidence"), code)
        if (stage_id in parsed_resolution_stages or
                stage_status_id not in {1, 3} or
                (stage_status_id == 1 and stage_reason_id != 0) or
                stage_backend_id != backend_id or
                stage_provider_id not in expected_providers or
                stage_adapter != adapter or
                not _same_domain(stage_domain, domain)):
            _fail("typed_status.island_resolution_stage_mismatch")
        parsed_resolution_stages[stage_id] = {
            "effect_id": effect_id, "effect": effect,
            "backend_id": stage_backend_id,
            "provider_id": stage_provider_id,
            "status_id": stage_status_id, "reason_id": stage_reason_id,
            "required": required, "failure_action_id": failure_action_id,
        }

    readiness = _object(item.get("readiness"), code)
    readiness_program_id = _uint(
        readiness.get("program_id"), code, nonzero=True)
    if not _boolean(readiness.get("configured"), code):
        _fail(code)
    readiness_generations = _invalidation_generations(
        readiness.get("invalidation_key"), code)
    if readiness_generations != program_generations:
        _fail("typed_status.stale_generation")
    readiness_status_id, _ = _pair(
        readiness.get("status"), STATUSES, code)
    readiness_reason_id, _ = _pair(
        readiness.get("reason"), REASONS, code)
    readiness_requested_backend_id, _ = _pair(
        readiness.get("requested_backend"), BACKENDS, code)
    readiness_backend_id, _ = _pair(
        readiness.get("selected_backend"), BACKENDS, code)
    readiness_provider_id, _ = _pair(readiness.get("provider"), PROVIDERS, code)
    readiness_domain = _domain(readiness.get("execution_domain"), code)
    _uint(readiness.get("ready_stage_mask"), code)
    _uint(readiness.get("blocked_stage_mask"), code)
    _uint(readiness.get("disabled_stage_mask"), code)
    readiness_degraded = _boolean(readiness.get("degraded"), code)
    _boolean(readiness.get("rebuild_requested"), code)
    _semantic_evidence(readiness.get("evidence"), code)
    if (readiness_status_id not in {1, 3} or
            readiness_degraded != (readiness_status_id == 3) or
            (readiness_status_id == 1 and readiness_reason_id != 0) or
            readiness_requested_backend_id not in {0, backend_id} or
            readiness_backend_id != backend_id or
            readiness_provider_id != provider_id or
            not _same_domain(readiness_domain, domain) or
            not _boolean(readiness.get("available"), code)):
        _fail("typed_status.island_readiness_mismatch")

    telemetry = _object(item.get("telemetry"), code)
    telemetry_backend_id, _ = _pair(
        telemetry.get("selected_backend"), BACKENDS, code)
    telemetry_requested_backend_id, _ = _pair(
        telemetry.get("requested_backend"), BACKENDS, code)
    telemetry_active_backend_id, _ = _pair(
        telemetry.get("active_backend"), BACKENDS, code)
    telemetry_active_provider_id, _ = _pair(
        telemetry.get("active_provider"), PROVIDERS, code)
    telemetry_runtime_status_id, telemetry_runtime_status = _pair(
        telemetry.get("runtime_status"), STATUSES, code)
    _pair(telemetry.get("runtime_reason"), REASONS, code)
    if (telemetry_requested_backend_id not in {0, backend_id} or
            telemetry_backend_id != backend_id or
            telemetry_active_backend_id not in {0, backend_id} or
            (telemetry_active_backend_id == 0) !=
            (telemetry_active_provider_id == 0) or
            (telemetry_active_provider_id != 0 and
             telemetry_active_provider_id not in expected_providers)):
        _fail("typed_status.island_telemetry_mismatch")
    transfers = _transfers(telemetry.get("transfers"), code)

    stages = _array(item.get("stages"), code, MAX_STAGES)
    if len(stages) != len(resolution_stages) or not stages:
        _fail("typed_status.island_stage_mismatch")
    parsed_stages: dict[int, dict[str, Any]] = {}
    for raw_stage in stages:
        stage = _object(raw_stage, code)
        stage_id = _uint(stage.get("stage_id"), code, nonzero=True)
        effect_id, effect = _pair(stage.get("effect"), EFFECT_IDENTITIES, code)
        stage_backend_id, _ = _pair(stage.get("backend"), BACKENDS, code)
        stage_provider_id, _ = _pair(stage.get("provider"), PROVIDERS, code)
        stage_status_id, _ = _pair(stage.get("status"), STATUSES, code)
        stage_reason_id, _ = _pair(stage.get("reason"), REASONS, code)
        failure_action_id, _ = _pair(
            stage.get("failure_action"), FAILURE_ACTIONS, code)
        committed_id, committed_status = _stage_status(
            stage.get("committed_status"), code)
        required = _boolean(stage.get("required"), code)
        resolved = _boolean(stage.get("resolved"), code)
        committed = _boolean(stage.get("committed"), code)
        resolution_stage = parsed_resolution_stages.get(stage_id)
        if (stage_id in parsed_stages or stage_backend_id != backend_id or
                stage_provider_id not in expected_providers or
                not resolved or resolution_stage is None or
                resolution_stage["effect_id"] != effect_id or
                resolution_stage["backend_id"] != stage_backend_id or
                resolution_stage["provider_id"] != stage_provider_id or
                resolution_stage["status_id"] != stage_status_id or
                resolution_stage["reason_id"] != stage_reason_id or
                resolution_stage["required"] != required or
                resolution_stage["failure_action_id"] != failure_action_id or
                committed != (committed_id != 0)):
            _fail("typed_status.island_stage_mismatch")
        parsed_stages[stage_id] = {
            "effect": effect, "effect_id": effect_id,
            "provider_id": stage_provider_id,
            "committed": committed, "committed_status": committed_status,
            "committed_status_id": committed_id,
        }
    if parsed_stages.keys() != parsed_resolution_stages.keys():
        _fail("typed_status.island_stage_mismatch")

    committed = _object(item.get("latest_committed_frame"), code)
    committed_program_id = _uint(committed.get("program_id"), code)
    committed_identity = {
        "valid": _boolean(committed.get("valid"), code),
        "capture_sequence": _uint(committed.get("capture_sequence"), code),
        "output_commit_sequence": _uint(
            committed.get("output_commit_sequence"), code),
    }
    if committed_identity["valid"]:
        if committed_identity != writer_commit:
            _fail("typed_status.mixed_writer_commit")
        if committed_program_id != readiness_program_id:
            _fail("typed_status.commit_program_mismatch")
        commit_generations = _invalidation_generations(
            committed.get("invalidation_key"), code)
        commit_domain = _domain(committed.get("execution_domain"), code)
        if (commit_generations != program_generations or
                not _same_domain(commit_domain, domain)):
            _fail("typed_status.commit_binding_mismatch")
    elif (committed_program_id or committed_identity["capture_sequence"] or
          committed_identity["output_commit_sequence"]):
        _fail("typed_status.invalid_commit")
    committed_stages = _array(committed.get("stages"), code, MAX_STAGES)
    if not committed_identity["valid"] and committed_stages:
        _fail("typed_status.invalid_commit")
    seen_committed_stages: set[int] = set()
    for raw_stage in committed_stages:
        stage = _object(raw_stage, code)
        stage_id = _uint(stage.get("stage_id"), code, nonzero=True)
        effect_id, effect = _pair(stage.get("effect"), EFFECT_IDENTITIES, code)
        stage_backend_id, _ = _pair(stage.get("backend"), BACKENDS, code)
        stage_provider_id, _ = _pair(stage.get("provider"), PROVIDERS, code)
        status_id, status = _stage_status(stage.get("status"), code)
        _pair(stage.get("reason"), REASONS, code)
        known = parsed_stages.get(stage_id)
        if (stage_id in seen_committed_stages or known is None or
                known["effect_id"] != effect_id or
                known["provider_id"] != stage_provider_id or
                stage_backend_id != backend_id or
                known["committed_status_id"] != status_id):
            _fail("typed_status.commit_stage_mismatch")
        seen_committed_stages.add(stage_id)
        known["commit_frame_status"] = status
    return ({
        "route": route, "route_id": route_id, "island_id": island_id,
        "adapter_instance_id": adapter, "backend": backend,
        "backend_id": backend_id, "provider": provider,
        "provider_id": provider_id, "execution_domain": domain,
        "transfers": transfers, "current": True,
        "writer_commit": committed_identity,
        "runtime_status": telemetry_runtime_status,
        "runtime_status_id": telemetry_runtime_status_id,
        "telemetry_active_backend_id": telemetry_active_backend_id,
        "telemetry_active_provider_id": telemetry_active_provider_id,
    }, parsed_stages)


def _parse_effect(
    value: Any,
    lifecycle: str,
    writer_commit: dict[str, Any],
    islands: dict[int, dict[str, Any]],
    island_stages: dict[int, dict[int, dict[str, Any]]],
    capability_lanes: dict[tuple[str, int], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    code = "typed_status.effect_invalid"
    item = _object(value, code)
    effect_id, effect = _pair(item.get("effect"), EFFECT_IDENTITIES, code)
    requested_capability = _capability_selection(
        item.get("requested_capability"), effect, code)
    selected_capability = _capability_selection(
        item.get("selected_capability"), effect, code)
    requested_backend_id, requested_backend = _pair(
        item.get("requested_backend"), BACKENDS, code)
    requested_provider_id, requested_provider = _pair(
        item.get("requested_provider"), PROVIDERS, code)
    selected_backend_id, selected_backend = _pair(
        item.get("selected_backend"), BACKENDS, code)
    selected_provider_id, selected_provider = _pair(
        item.get("selected_provider"), PROVIDERS, code)
    active_backend_id, active_backend = _pair(
        item.get("active_backend"), BACKENDS, code)
    active_provider_id, active_provider = _pair(
        item.get("active_provider"), PROVIDERS, code)
    status_id, status = _pair(item.get("status"), STATUSES, code)
    reason_id, reason = _pair(item.get("reason"), REASONS, code)
    _pair(item.get("failure_action"), FAILURE_ACTIONS, code)
    commit = _commit(item.get("commit"), code)
    selected_island_id = _uint(item.get("selected_island_id"), code)
    committed_stage_id = _uint(item.get("committed_stage_id"), code)
    committed_status_id, committed_status = _stage_status(
        item.get("committed_stage_status"), code)
    strict_backend = _boolean(item.get("strict_backend_selection"), code)
    strict_provider = _boolean(item.get("strict_provider_selection"), code)
    flags = {
        key: _boolean(item.get(key), code)
        for key in (
            "required", "requested", "available", "selected", "active",
            "degraded", "blocked", "fallback",
        )
    }
    backend_lanes = {1: 4, 2: 2, 3: 1, 4: 2, 5: 3}
    if flags["requested"]:
        requested_lane = capability_lanes.get(
            (effect, requested_capability["lane_id"]))
        if (requested_capability["lane_id"] == 0 or
                backend_lanes.get(requested_backend_id) !=
                requested_capability["lane_id"] or
                requested_lane is None or
                not _capability_matches(requested_capability, requested_lane)):
            _fail("typed_status.requested_capability_mismatch")
        requested_capability["disposition"] = requested_lane["disposition"]
        requested_capability["selectable"] = requested_lane["selectable"]
        requested_eligible = (
            requested_lane["selectable"] and
            (requested_provider_id == 0 or
             requested_provider_id == requested_lane["provider_id"]))
        if not flags["selected"] and flags["available"] != requested_eligible:
            _fail("typed_status.requested_capability_mismatch")
    elif (requested_capability["lane_id"] != 0 or flags["available"] or
          flags["selected"] or flags["active"] or flags["degraded"] or
          flags["blocked"] or flags["fallback"] or flags["required"] or
          strict_backend or strict_provider or requested_backend_id != 0 or
          requested_provider_id != 0 or selected_backend_id != 0 or
          selected_provider_id != 0 or selected_island_id != 0 or
          committed_stage_id != 0 or committed_status_id != 0 or
          commit["valid"]):
        _fail(code)
    if flags["selected"]:
        selected_lane = capability_lanes.get(
            (effect, selected_capability["lane_id"]))
        if (backend_lanes.get(selected_backend_id) !=
                selected_capability["lane_id"] or
                selected_lane is None or not selected_lane["selectable"] or
                selected_capability["provider_id"] != selected_provider_id or
                not _capability_matches(selected_capability, selected_lane) or
                not _selected_capability_is_exact(
                    effect, selected_capability)):
            _fail("typed_status.selected_capability_incomplete")
        selected_capability["disposition"] = selected_lane["disposition"]
        selected_capability["selectable"] = selected_lane["selectable"]
    elif selected_capability["lane_id"] != 0:
        _fail(code)
    if strict_backend and flags["selected"] and requested_backend_id != selected_backend_id:
        _fail("typed_status.strict_backend_substitution")
    if strict_provider and flags["selected"] and requested_provider_id != selected_provider_id:
        _fail("typed_status.strict_provider_substitution")
    if flags["blocked"] and (
            flags["selected"] or flags["active"] or reason_id == 0 or
            status_id not in {2, 4, 8, 10}):
        _fail(code)
    if flags["degraded"] != (status_id == 3):
        _fail(code)
    if not flags["selected"] and (
            selected_backend_id != 0 or selected_provider_id != 0 or
            selected_island_id != 0):
        _fail(code)
    island = islands.get(selected_island_id)
    stage = island_stages.get(selected_island_id, {}).get(committed_stage_id)
    if commit["valid"]:
        if (commit != writer_commit or not flags["selected"] or
                island is None or stage is None or
                island["backend_id"] != selected_backend_id or
                stage["effect_id"] != effect_id or
                stage["provider_id"] != selected_provider_id or
                not stage["committed"] or
                stage["committed_status_id"] != committed_status_id or
                stage.get("commit_frame_status") != committed_status):
            _fail("typed_status.commit_activity_mismatch")
    elif committed_stage_id != 0 or committed_status_id != 0:
        _fail("typed_status.commit_activity_mismatch")
    committed_success = (
        commit["valid"] and committed_status_id == 1 and
        lifecycle == "running" and flags["selected"] and flags["available"] and
        island is not None and island["runtime_status_id"] in {1, 3, 6} and
        island["telemetry_active_backend_id"] == selected_backend_id and
        island["telemetry_active_provider_id"] == selected_provider_id)
    if flags["active"] != committed_success:
        _fail("typed_status.active_attribution_invalid")
    if flags["active"]:
        if (active_backend_id == 0 or active_provider_id == 0 or
                active_backend_id != selected_backend_id or
                active_provider_id != selected_provider_id or
                status_id != 6 or reason_id != 0 or flags["blocked"]):
            _fail("typed_status.active_attribution_invalid")
    elif active_backend_id != 0 or active_provider_id != 0:
        _fail("typed_status.inactive_attribution_invalid")
    optional_failure = commit["valid"] and committed_status_id == 2
    if optional_failure and (
            flags["required"] or flags["active"] or not flags["degraded"] or
            not flags["fallback"] or status_id != 3 or reason_id != 12):
        _fail("typed_status.optional_failure_inconsistent")
    substitution = (
        flags["selected"] and
        (selected_backend_id != requested_backend_id or
         (requested_provider_id != 0 and
          selected_provider_id != requested_provider_id)))
    if flags["fallback"] != (substitution or optional_failure):
        _fail(code)
    if flags["selected"] and selected_island_id not in islands:
        _fail("typed_status.selected_island_missing")
    return effect, {
        "effect_id": effect_id,
        "requested_backend": requested_backend,
        "requested_provider": requested_provider,
        "selected_backend": selected_backend,
        "selected_provider": selected_provider,
        "active_backend": active_backend,
        "active_provider": active_provider,
        "status": status, "status_id": status_id,
        "reason": reason, "reason_id": reason_id,
        "requested_capability": requested_capability,
        "selected_capability": selected_capability,
        "commit": commit, "selected_island_id": selected_island_id,
        "committed_stage_id": committed_stage_id,
        "committed_stage_status": committed_status,
        **flags,
    }


def _invalid_result(present: bool, code: str) -> dict[str, Any]:
    return {
        "present": present,
        "valid": False,
        "mode": "shadow_only",
        "recommendation_eligible": False,
        "reason_codes": [code],
        "islands": [],
        "effects": {},
    }


def parse_typed_multi_island_video_status(
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize optional daemon evidence without promoting it."""
    if not isinstance(diagnostics, dict):
        return _invalid_result(False, "typed_status.absent")
    if "typed_multi_island_video_status" not in diagnostics:
        return _invalid_result(False, "typed_status.absent")
    try:
        value = _object(
            diagnostics.get("typed_multi_island_video_status"),
            "typed_status.root_invalid")
        if value.get("valid") is not True:
            _fail("typed_status.root_invalid")
        if value.get("schema_version") != SCHEMA_VERSION:
            _fail("typed_status.unsupported_schema_version")
        if value.get("hal_contract_version") != HAL_CONTRACT_VERSION:
            _fail("typed_status.unsupported_hal_contract_version")
        if value.get("reason_code_version") != REASON_CODE_VERSION:
            _fail("typed_status.unsupported_reason_code_version")
        status_generation = _uint(
            value.get("status_generation"), "typed_status.invalid_generation",
            nonzero=True)
        generations = _generations(
            value.get("generations"), GENERATION_KEYS,
            "typed_status.invalid_generation")
        program_generations = _generations(
            value.get("program_generations"), PROGRAM_GENERATION_KEYS,
            "typed_status.invalid_generation")
        for key in ("configuration", "provider", "device", "context"):
            if generations[key] != program_generations[key]:
                _fail("typed_status.mixed_generation")
        authority = _object(
            value.get("authority"), "typed_status.authority_invalid")
        if authority.get("scope") != "prepared_routes_only":
            _fail("typed_status.authority_invalid")
        authority_complete = _boolean(
            authority.get("complete"), "typed_status.authority_invalid")
        prepared_effect_mask = _uint(
            authority.get("prepared_route_effect_mask"),
            "typed_status.authority_invalid")
        uncovered_effect_mask = _uint(
            authority.get("compatibility_uncovered_effect_mask"),
            "typed_status.authority_invalid")
        canonical_effect_mask = (1 << len(EFFECT_IDS)) - 1
        if ((prepared_effect_mask | uncovered_effect_mask) &
                ~canonical_effect_mask or
                prepared_effect_mask & uncovered_effect_mask or
                authority_complete != (uncovered_effect_mask == 0)):
            _fail("typed_status.authority_invalid")
        if not authority_complete:
            _fail("typed_status.authority_incomplete")
        _, lifecycle = _pair(
            value.get("lifecycle"), LIFECYCLES, "typed_status.lifecycle_invalid")
        writer_commit = _commit(
            value.get("writer_commit"), "typed_status.invalid_commit")

        capabilities = _object(
            value.get("capabilities"), "typed_status.capabilities_invalid")
        if (capabilities.get("valid") is not True or
                capabilities.get("schema_version") != SCHEMA_VERSION or
                capabilities.get("hal_contract_version") != HAL_CONTRACT_VERSION or
                capabilities.get("reason_code_version") != REASON_CODE_VERSION):
            _fail("typed_status.capabilities_invalid")
        capability_generations = _generations(
            capabilities.get("generations"), GENERATION_KEYS,
            "typed_status.capabilities_invalid")
        if capability_generations != generations:
            _fail("typed_status.mixed_generation")
        refresh = _object(
            capabilities.get("refresh"), "typed_status.capabilities_invalid")
        if (refresh.get("status") != "succeeded" or
                refresh.get("current") is not True or
                refresh.get("requested_generation") != generations["capability"] or
                refresh.get("completed_generation") != generations["capability"]):
            _fail("typed_status.capability_snapshot_stale")
        refresh_code = _capability_code(
            refresh.get("code"), "typed_status.capabilities_invalid")
        if refresh_code["id"] != 200:
            _fail("typed_status.capability_snapshot_stale")
        capability_lanes = _capability_matrix(
            capabilities, "typed_status.capabilities_invalid")

        raw_islands = _array(
            value.get("islands"), "typed_status.island_count_invalid",
            MAX_ISLANDS)
        islands: dict[int, dict[str, Any]] = {}
        island_stages: dict[int, dict[int, dict[str, Any]]] = {}
        aggregate_sum = {key: 0 for key in TRANSFER_KEYS}
        adapter_ids: set[int] = set()
        for raw_island in raw_islands:
            island, stages = _parse_island(
                raw_island, status_generation, generations["capability"],
                program_generations, writer_commit)
            island_id = island["island_id"]
            adapter_id = island["adapter_instance_id"]
            if island_id in islands or adapter_id in adapter_ids:
                _fail("typed_status.duplicate_island")
            islands[island_id] = island
            island_stages[island_id] = stages
            adapter_ids.add(adapter_id)
            for key in TRANSFER_KEYS:
                aggregate_sum[key] += island["transfers"][key]

        raw_effects = _array(
            value.get("effects"), "typed_status.effect_count_invalid",
            len(EFFECT_IDS))
        if len(raw_effects) != len(EFFECT_IDS):
            _fail("typed_status.effect_count_invalid")
        effects: dict[str, dict[str, Any]] = {}
        for raw_effect in raw_effects:
            effect, normalized = _parse_effect(
                raw_effect, lifecycle, writer_commit, islands, island_stages,
                capability_lanes)
            if effect in effects:
                _fail("typed_status.duplicate_effect")
            effects[effect] = normalized
        if set(effects) != set(EFFECT_IDS) or set(effects) != set(CANONICAL_EFFECTS):
            _fail("typed_status.unknown_or_missing_effect")
        expected_prepared_effect_mask = sum(
            1 << (EFFECT_IDS[effect] - 1)
            for effect, state in effects.items() if state["selected"])
        expected_uncovered_effect_mask = sum(
            1 << (EFFECT_IDS[effect] - 1)
            for effect, state in effects.items()
            if state["requested"] and not state["selected"])
        if (prepared_effect_mask != expected_prepared_effect_mask or
                uncovered_effect_mask != expected_uncovered_effect_mask):
            _fail("typed_status.authority_invalid")
        if (any(effect["active"] for effect in effects.values()) and
                not writer_commit["valid"]):
            _fail("typed_status.writer_commit_activity_mismatch")

        aggregate = _object(
            value.get("aggregate"), "typed_status.aggregate_invalid")
        if (_uint(aggregate.get("current_island_count"),
                  "typed_status.aggregate_invalid") != len(islands) or
                _transfers(aggregate.get("transfers"),
                           "typed_status.aggregate_invalid") != aggregate_sum):
            _fail("typed_status.aggregate_mismatch")
        return {
            "present": True,
            "valid": True,
            "mode": "shadow_only",
            "recommendation_eligible": False,
            "reason_codes": ["typed_status.valid_shadow_evidence"],
            "schema_version": SCHEMA_VERSION,
            "hal_contract_version": HAL_CONTRACT_VERSION,
            "reason_code_version": REASON_CODE_VERSION,
            "status_generation": status_generation,
            "lifecycle": lifecycle,
            "generations": generations,
            "program_generations": program_generations,
            "writer_commit": writer_commit,
            "authority": {
                "scope": "prepared_routes_only",
                "complete": True,
                "prepared_route_effect_mask": prepared_effect_mask,
                "compatibility_uncovered_effect_mask": 0,
            },
            "islands": [islands[key] for key in sorted(islands)],
            "effects": effects,
            "aggregate": {
                "current_island_count": len(islands),
                "transfers": aggregate_sum,
            },
        }
    except _InvalidStatus as exc:
        return _invalid_result(True, exc.code)
