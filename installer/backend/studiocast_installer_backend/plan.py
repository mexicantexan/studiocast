"""Plan construction, persistence, and approval contracts."""

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
    BINARIES,
    BackendError,
    DEFAULT_MODEL_PACKS,
    PLAN_VERSION,
    POLICY_VERSION,
    V4L_OWNED_PATHS,
    atomic_write,
    digest,
    installer_version,
    parse_time,
    read_version,
    roots,
    utc_now,
    validate_mutation_path,
)

from .packaged import (
    catalog_size_summary,
    load_model_catalog,
    release_cache_dir,
    release_module,
    validated_catalog_size_summary,
)

def operation(op_id: str, kind: str, depends: list[str], inputs: dict[str, Any],
              category: str, privilege: str = "user", failure: str = "abort_rollback",
              source_paths: list[str] | None = None, target_paths: list[str] | None = None) -> dict[str, Any]:
    return {"id": op_id, "kind": kind, "depends_on": depends, "inputs": inputs,
        "source_paths": source_paths or [], "target_paths": target_paths or [],
        "privilege": privilege, "authorization_operation": inputs.get("operation_type") if privilege == "trusted_helper" else None,
        "reversibility": "reversible" if kind not in {"cleanup"} else "irreversible",
        "failure_policy": failure, "preconditions": [], "validation": [],
        "review": {"category": category, "message_code": "plan." + op_id}}


def desired_from_args(args: argparse.Namespace, recommendation: dict[str, Any]) -> dict[str, Any]:
    desired = copy.deepcopy(recommendation["selections"])
    if getattr(args, "build_type", None): desired["build_type"] = args.build_type
    if getattr(args, "service", None) is not None: desired["service"] = {"desired": "enabled_started" if args.service else "disabled"}
    if getattr(args, "v4l2", None) is not None:
        desired["virtual_camera"]["desired"] = args.v4l2
        desired["virtual_camera"]["required_for_success"] = bool(
            args.v4l2 and recommendation.get("selection_route", recommendation.get("route")) == "recommended")
    if getattr(args, "load_loopback", None) is not None: desired["virtual_camera"]["load"] = args.load_loopback
    if getattr(args, "persist_loopback", None) is not None: desired["virtual_camera"]["persist"] = args.persist_loopback
    if getattr(args, "deps", None) is not None: desired["manage_dependencies"] = args.deps
    if getattr(args, "open_backends", None) is not None:
        desired.setdefault("features", {})["open_cuda"] = args.open_backends
        desired["features"]["open_audio"] = args.open_backends
    if getattr(args, "open_cuda", None) is not None:
        desired.setdefault("features", {})["open_cuda"] = args.open_cuda
    if getattr(args, "open_audio", None) is not None:
        desired.setdefault("features", {})["open_audio"] = args.open_audio
    if getattr(args, "open_vulkan", None) is not None: desired.setdefault("features", {})["open_vulkan"] = args.open_vulkan
    if getattr(args, "models", None) is not None: desired["model_pack_ids"] = list(DEFAULT_MODEL_PACKS) if args.models else []
    if getattr(args, "model_ids", None): desired["model_pack_ids"] = list(dict.fromkeys(args.model_ids))
    if getattr(args, "model_destination", None): desired["model_destination"] = str(Path(args.model_destination).absolute())
    if getattr(args, "v4l_device_number", None) is not None: desired["virtual_camera"]["device_number"] = args.v4l_device_number
    if getattr(args, "v4l_label", None) is not None: desired["virtual_camera"]["label"] = args.v4l_label
    if getattr(args, "v4l_exclusive_caps", None) is not None: desired["virtual_camera"]["exclusive_caps"] = args.v4l_exclusive_caps
    if getattr(args, "vulkan_runtime", False): desired["vulkan_runtime"] = True
    if getattr(args, "mesa_vulkan", False): desired["mesa_packages"] = True
    if getattr(args, "shader_tools", False): desired["shader_tools"] = True
    if getattr(args, "preserve_settings", None) is not None: desired["preserve_settings"] = args.preserve_settings
    desired.setdefault("features", {}).setdefault("open_cuda", True)
    desired["features"].setdefault("open_audio", True)
    desired["features"].setdefault("open_vulkan", False)
    return desired


def plan_digest(plan: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(plan)
    unsigned.pop("plan_digest", None)
    unsigned.pop("approval_token", None)
    return digest(unsigned)


def pending_path(plan_id: str) -> Path:
    return roots()["state"] / "installer/pending-plans" / f"{plan_id}.json"


def persist_pending(plan: dict[str, Any]) -> None:
    atomic_write(pending_path(plan["plan_id"]), {"plan_id": plan["plan_id"], "digest": plan["plan_digest"],
        "token": plan["approval_token"], "uid": os.getuid(), "expires_at": plan["expires_at"], "used": False})


def source_for_plan(args: argparse.Namespace, target: str) -> tuple[str, str]:
    if getattr(args, "release_archive", None):
        return "archive", str(Path(args.release_archive).absolute())
    if getattr(args, "source_dir", None):
        return "directory", str(Path(args.source_dir).absolute())
    # Installed AppImages normally pass their verified bundled archive.  The
    # fallback is useful only to developer CLI callers.
    repo = Path(__file__).resolve().parents[3]
    return "directory", str(repo)


def create_plan(args: argparse.Namespace, facts: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    r = roots()
    desired = desired_from_args(args, rec)
    camera_contract = desired.get("virtual_camera", {})
    if (type(camera_contract.get("device_number", 10)) is not int or
            not 0 <= camera_contract.get("device_number", 10) <= 63 or
            type(camera_contract.get("exclusive_caps", True)) is not bool or
            re.fullmatch(r"[ A-Za-z0-9_.()+-]{1,64}",
                         str(camera_contract.get("label", "StudioCast Camera"))) is None):
        raise BackendError("v4l.invalid_configuration", "Advanced virtual-camera arguments are invalid")
    intent = args.workflow
    if intent in {"clean-install", "reinstall"}:
        desired["model_pack_ids"] = list(DEFAULT_MODEL_PACKS)
    target = getattr(args, "target_version", None) or facts.get("installation", {}).get("target_version") or read_version(Path(getattr(args, "source_dir", None) or ""))
    if target == "unknown": target = facts.get("installation", {}).get("active_version") or "unknown"
    source_kind, source_path = source_for_plan(args, target)
    verified_release = copy.deepcopy(getattr(args, "verified_release", None))
    if verified_release:
        source_kind, source_path = "archive", verified_release["archive_path"]
    model_packs: dict[str, dict[str, Any]] = {}
    model_catalog: dict[str, Any] | None = None
    model_catalog_sha256: str | None = None
    model_source: str | None = None
    model_destination = Path(desired.get("model_destination") or r["data"] / "models").absolute()
    validate_mutation_path(model_destination, [r["data"] / "models"], allow_root=True)
    desired["model_destination"] = str(model_destination)
    if desired.get("model_pack_ids"):
        _, model_catalog, _, model_catalog_sha256 = load_model_catalog()
        model_packs = {pack["id"]: pack for pack in model_catalog["packs"]}
        if any(pack not in model_packs for pack in desired["model_pack_ids"]):
            raise BackendError("model.catalog_missing_default", "The packaged catalog omits a selected model pack")
        requested_source = getattr(args, "model_source", None)
        if (requested_source and rec.get("selection_route", rec["route"]) != "advanced" and
                os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") != "1"):
            raise BackendError("model.source.advanced_only", "Custom model sources are available only in Advanced mode")
        model_source = requested_source or desired.get("model_source") or model_catalog["default_source"]
    model_offline = bool(getattr(args, "offline", False) or
                         os.environ.get("STUDIOCAST_INSTALLER_OFFLINE") == "1" or
                         desired.get("model_offline", False))
    release_offline = bool(getattr(args, "offline", False) or
                           os.environ.get("STUDIOCAST_INSTALLER_OFFLINE") == "1")
    desired["model_source"] = model_source
    desired["model_offline"] = model_offline
    build_cache = Path(getattr(args, "build_dir", "") or r["cache"] / "builds" / target).absolute()
    payload_root = r["data"] / "payloads"
    plan_id = str(uuid.uuid4())
    staging = payload_root / f".staging-{plan_id}"
    target_payload = payload_root / target
    current = r["data"] / "current"
    ops: list[dict[str, Any]] = [operation("preflight.validate", "preflight", [], {}, "validation", "none")]
    last = "preflight.validate"
    plan_specific_warnings: list[dict[str, Any]] = []
    preservation = {"preserve_settings": bool(desired.get("preserve_settings", True)), "snapshot_path": None}

    def append_model_transaction(pack_id: str) -> None:
        nonlocal last
        pack = model_packs[pack_id]
        op_id = f"model.pack.{pack_id}.transaction"
        inputs = {"pack_id": pack_id, "family": pack["family"],
                  "catalog_version": model_catalog["catalog_version"],
                  "catalog_sha256": model_catalog_sha256,
                  "source": model_source, "offline": model_offline, "retries": 2,
                  "recommended": rec.get("selection_route", rec["route"]) == "recommended",
                  "destination_root": str(model_destination),
                  "pack": copy.deepcopy(pack)}
        ops.append(operation(op_id, "model_transaction", [last], inputs,
                             "downloads", failure="degrade"))
        last = op_id
    if intent in {"clean-install", "reinstall"}:
        if preservation["preserve_settings"]:
            # The clean-install state wipe must not erase its own backup.  Keep
            # the short-lived copy as a sibling of the StudioCast config root.
            snapshot = r["config"].parent / f".studiocast-settings-backup-{plan_id}"
            preservation["snapshot_path"] = str(snapshot)
            ops.append(operation("settings.snapshot", "settings_snapshot", [last], {"snapshot_path": str(snapshot)}, "preserved_items"))
            last = "settings.snapshot"
        for op_id, target_name in (("cleanup.payloads", "payloads"), ("cleanup.service", "service"),
                ("cleanup.build_cache", "build_cache"), ("cleanup.runtime_state", "runtime_state"),
                ("cleanup.models", "models")):
            ops.append(operation(op_id, "cleanup", [last], {"target": target_name}, "removed_items"))
            last = op_id
        # Clean means reset configuration too.  A selected preservation snapshot
        # is restored transactionally; otherwise this is a literal wipe.
        ops.append(operation("cleanup.settings", "cleanup", [last], {"target": "settings"}, "removed_items"))
        last = "cleanup.settings"
        desired["model_pack_ids"] = list(DEFAULT_MODEL_PACKS)
    minimal_repair = intent == "repair" and facts.get("installation", {}).get("classification") == "healthy"
    if intent == "uninstall":
        for op_id, target_name in (("service.state.reconcile", "service_state"), ("service.unit.reconcile", "service_unit"),
                ("links.reconcile", "links"), ("cleanup.payloads", "payloads")):
            kind = "service_state_reconcile" if target_name == "service_state" else "service_unit_reconcile" if target_name == "service_unit" else "links_reconcile" if target_name == "links" else "cleanup"
            ops.append(operation(op_id, kind, [last], {"desired": "absent", "target": target_name}, "application_files"))
            last = op_id
        owned = facts.get("v4l2", {}).get("owned_configuration", [])
        expected_hashes = {item.get("id"): item.get("sha256") for item in owned
                           if isinstance(item, dict) and item.get("id") in V4L_OWNED_PATHS}
        if set(expected_hashes) == set(V4L_OWNED_PATHS):
            ops.append(operation("v4l.persistence.remove", "trusted_helper", [last],
                                 {"operation_type": "v4l.persistence.remove.v1",
                                  "expected_hashes": expected_hashes},
                                 "system_changes", "trusted_helper", failure="degrade"))
            last = "v4l.persistence.remove"
        elif facts.get("v4l2", {}).get("persistence_state") != "absent":
            plan_specific_warnings.append({"code": "v4l.persistence.ownership_unverified_manual_cleanup",
                                           "applies_to": "system_changes"})
        if not preservation["preserve_settings"]:
            for op_id, target_name in (("cleanup.models", "models"), ("cleanup.settings", "settings"),
                    ("cleanup.runtime_state", "runtime_state"), ("cleanup.cache", "cache"),
                    ("cleanup.data", "data")):
                ops.append(operation(op_id, "cleanup", [last], {"target": target_name}, "removed_items")); last = op_id
        ops.append(operation("manifest.commit", "uninstall_tombstone", [last], {}, "application_files"))
    elif minimal_repair:
        v4l = desired["virtual_camera"]
        vf = facts.get("v4l2", {})
        if v4l.get("desired") is True:
            helper_args = {"device_number": int(v4l.get("device_number", 10)),
                           "label": str(v4l.get("label", "StudioCast Camera")),
                           "exclusive_caps": bool(v4l.get("exclusive_caps", True))}
            if not vf.get("module_available"):
                ops.append(operation("v4l.package.ensure", "trusted_helper", [last], {"operation_type": "packages.ensure.v1", "packages": ["v4l2loopback-dkms", "v4l2loopback-utils"]}, "system_changes", "trusted_helper")); last = "v4l.package.ensure"
            if v4l.get("load", True) and (not vf.get("module_loaded") or not vf.get("device_present")):
                ops.append(operation("v4l.module.load", "trusted_helper", [last], {"operation_type": "v4l.module.load.v1", **helper_args}, "system_changes", "trusted_helper")); last = "v4l.module.load"
            if v4l.get("persist") and vf.get("persistence_state") != "configured":
                ops.append(operation("v4l.persistence.write", "trusted_helper", [last], {"operation_type": "v4l.persistence.write.v1", **helper_args}, "system_changes", "trusted_helper")); last = "v4l.persistence.write"
        if not facts.get("installation", {}).get("links_healthy", True):
            ops.append(operation("links.reconcile", "links_reconcile", [last], {"desired": "present", "binaries": list(BINARIES)}, "application_files")); last = "links.reconcile"
        service_actual = facts.get("service", {}).get("actual", "unknown")
        service_desired = desired["service"]["desired"]
        expected_actual = "active" if service_desired == "enabled_started" else "inactive"
        if service_desired in {"enabled_started", "disabled"} and service_actual != expected_actual:
            ops.append(operation("service.unit.reconcile", "service_unit_reconcile", [last], {"desired": service_desired}, "service_changes")); last = "service.unit.reconcile"
            ops.append(operation("service.state.reconcile", "service_state_reconcile", [last], {"desired": service_desired}, "service_changes", failure="degrade")); last = "service.state.reconcile"
        pack_facts = facts.get("models", {}).get("packs", {})
        for pack in desired["model_pack_ids"]:
            if pack_facts.get(pack, {}).get("integrity") == "verified":
                continue
            append_model_transaction(pack)
        ops.append(operation("manifest.commit", "manifest_commit", [last], {}, "validation"))
    else:
        if verified_release:
            source_metadata = {"artifact_id": "studiocast-source",
                "filename": verified_release["archive_filename"],
                "url": verified_release["archive_url"],
                "size_bytes": verified_release["archive_size"],
                "sha256": verified_release["archive_sha256"],
                "signature": copy.deepcopy(verified_release["archive_signature"]),
                "provenance": copy.deepcopy(verified_release["provenance"]),
                "license": copy.deepcopy(verified_release["license"])}
            ops.append(operation("source.release.acquire", "release_acquire", [last], {
                "manifest_path": verified_release["manifest_path"],
                "signature_path": verified_release["signature_path"],
                "manifest_sha256": verified_release["manifest_sha256"],
                "manifest_key_id": verified_release["manifest_key_id"],
                "version": verified_release["version"],
                "artifact": source_metadata,
                "candidate_path": verified_release.get("archive_candidate_path"),
                "cache_dir": str(release_cache_dir(args) / "artifacts"),
                "offline": release_offline,
            }, "downloads", source_paths=[verified_release["manifest_path"],
                verified_release["signature_path"]] + ([verified_release["archive_candidate_path"]]
                if verified_release.get("archive_candidate_path") else []),
                target_paths=[source_path]))
            last = "source.release.acquire"
        if source_kind == "archive":
            extract_to = r["cache"] / "sources" / plan_id
            ops.append(operation("source.archive.extract", "source_extract", [last], {"archive": source_path, "destination": str(extract_to)}, "downloads", source_paths=[source_path], target_paths=[str(extract_to)]))
            last = "source.archive.extract"
        packages: list[str] = []
        missing_build_packages = [{"cmake": "cmake", "ninja": "ninja-build", "c++": "build-essential",
                                   "pkg-config": "pkg-config"}[name]
                                  for name in facts.get("toolchain", {}).get("missing_build_dependencies", [])
                                  if name in {"cmake", "ninja", "c++", "pkg-config"}]
        if desired.get("manage_dependencies"):
            packages += missing_build_packages
        if desired.get("vulkan_runtime"):
            packages += ["libvulkan1", "vulkan-tools"]
        if desired.get("mesa_packages"):
            packages += ["mesa-vulkan-drivers"]
        if desired.get("shader_tools"):
            packages += ["glslang-tools"]
        packages = list(dict.fromkeys(packages))
        if packages:
            ops.append(operation("dependencies.packages.ensure", "trusted_helper", [last],
                                 {"operation_type": "packages.ensure.v1", "packages": packages},
                                 "system_changes", "trusted_helper")); last = "dependencies.packages.ensure"
        ops.append(operation("build.cache.prepare", "build_cache_prepare", [last], {"fresh": bool(getattr(args, "fresh_build", False))}, "application_files", target_paths=[str(build_cache)])); last = "build.cache.prepare"
        cmake = {"STUDIOCAST_ENABLE_OPEN_CUDA": bool(desired["features"]["open_cuda"]),
                 "STUDIOCAST_ENABLE_OPEN_AUDIO": bool(desired["features"]["open_audio"]),
                 "STUDIOCAST_ENABLE_OPEN_VULKAN": bool(desired["features"]["open_vulkan"])}
        ops.append(operation("build.configure", "build_configure", [last], {"source_kind": source_kind, "source_path": source_path,
            "build_path": str(build_cache), "build_type": desired["build_type"], "cmake": cmake}, "application_files", source_paths=[source_path], target_paths=[str(build_cache)])); last = "build.configure"
        ops.append(operation("build.compile", "build_compile", [last], {"build_path": str(build_cache), "targets": list(BINARIES)}, "application_files")); last = "build.compile"
        ops.append(operation("payload.stage", "payload_stage", [last], {"version": target, "build_path": str(build_cache)}, "application_files", target_paths=[str(staging)])); last = "payload.stage"
        ops.append(operation("payload.validate", "payload_validate", [last], {"required_binaries": list(BINARIES)}, "validation", "none")); last = "payload.validate"
        v4l = desired["virtual_camera"]
        if v4l.get("desired") is True:
            vf = facts.get("v4l2", {})
            helper_args = {"device_number": int(v4l.get("device_number", 10)), "label": str(v4l.get("label", "StudioCast Camera")),
                           "exclusive_caps": bool(v4l.get("exclusive_caps", True))}
            if not vf.get("module_available"):
                ops.append(operation("v4l.package.ensure", "trusted_helper", [last], {"operation_type": "packages.ensure.v1", "packages": ["v4l2loopback-dkms", "v4l2loopback-utils"]}, "system_changes", "trusted_helper")); last = "v4l.package.ensure"
            if v4l.get("load", True) and (not vf.get("module_loaded") or not vf.get("device_present")):
                ops.append(operation("v4l.module.load", "trusted_helper", [last], {"operation_type": "v4l.module.load.v1", **helper_args}, "system_changes", "trusted_helper")); last = "v4l.module.load"
            if v4l.get("persist") and vf.get("persistence_state") != "configured":
                ops.append(operation("v4l.persistence.write", "trusted_helper", [last], {"operation_type": "v4l.persistence.write.v1", **helper_args}, "system_changes", "trusted_helper")); last = "v4l.persistence.write"
            elif not v4l.get("persist") and vf.get("persistence_state") == "configured":
                owned = vf.get("owned_configuration", [])
                expected_hashes = {item.get("id"): item.get("sha256") for item in owned
                                   if isinstance(item, dict) and item.get("id") in V4L_OWNED_PATHS}
                if set(expected_hashes) == set(V4L_OWNED_PATHS):
                    ops.append(operation("v4l.persistence.remove", "trusted_helper", [last],
                        {"operation_type": "v4l.persistence.remove.v1", "expected_hashes": expected_hashes},
                        "system_changes", "trusted_helper")); last = "v4l.persistence.remove"
                else:
                    plan_specific_warnings.append({"code": "v4l.persistence.ownership_unverified_manual_cleanup",
                                                   "applies_to": "system_changes"})
        ops.append(operation("payload.activate", "payload_activate", [last], {"version": target}, "application_files", target_paths=[str(target_payload), str(current)])); last = "payload.activate"
        ops.append(operation("links.reconcile", "links_reconcile", [last], {"desired": "present", "binaries": list(BINARIES)}, "application_files")); last = "links.reconcile"
        if desired["service"].get("desired") in {"enabled_started", "disabled"}:
            ops.append(operation("service.unit.reconcile", "service_unit_reconcile", [last], {"desired": desired["service"]["desired"]}, "service_changes")); last = "service.unit.reconcile"
            ops.append(operation("service.state.reconcile", "service_state_reconcile", [last], {"desired": desired["service"]["desired"]}, "service_changes", failure="degrade")); last = "service.state.reconcile"
        for pack in desired["model_pack_ids"]:
            append_model_transaction(pack)
        if intent in {"clean-install", "reinstall"} and preservation["preserve_settings"]:
            ops.append(operation("settings.restore", "settings_restore", [last], {"snapshot_path": preservation["snapshot_path"]}, "preserved_items")); last = "settings.restore"
        ops.append(operation("manifest.commit", "manifest_commit", [last], {}, "validation"))
    blockers = copy.deepcopy(rec["blockers"])
    warnings = copy.deepcopy(rec["warnings"]) + plan_specific_warnings
    if intent == "uninstall":
        # User-local removal remains available on unsupported/offline/low-disk
        # hosts and never needs a release source or build cache.
        blockers = []
    if not desired["virtual_camera"].get("desired"):
        blockers = [item for item in blockers if item.get("applies_to") != "virtual_camera"]
    if desired.get("service", {}).get("desired") in {"enabled_started", "disabled"}:
        blockers = [item for item in blockers
                    if item.get("code") != "configuration.service.selection_required"]
    if type(desired.get("virtual_camera", {}).get("desired")) is bool:
        blockers = [item for item in blockers
                    if item.get("code") != "configuration.virtual_camera.selection_required"]
    if intent != "uninstall" and rec.get("selection_route", rec["route"]) == "recommended" and desired.get("model_pack_ids"):
        declared_summary = validated_catalog_size_summary(facts)
        packaged_summary = catalog_size_summary(model_catalog, model_catalog_sha256)
        local_sizes_trusted = (
            model_catalog.get("size_metadata", {}).get("trusted", False) is True and
            bool(model_catalog.get("size_metadata", {}).get("reason_code")) and
            all(type(artifact.get("size_bytes")) is int and artifact["size_bytes"] > 0 and
                artifact.get("size_status") == "known"
                for pack_id in desired["model_pack_ids"]
                for artifact in model_packs[pack_id]["artifacts"]))
        if (not local_sizes_trusted or declared_summary != packaged_summary) and not any(
                item.get("code") == "artifact_sizes_untrusted" for item in blockers):
            blockers.append({"code": "artifact_sizes_untrusted", "applies_to": "models"})
    missing_build = facts.get("toolchain", {}).get("missing_build_dependencies", [])
    if missing_build and not desired.get("manage_dependencies"):
        blockers.append({"code": "toolchain.build_dependencies_missing", "applies_to": "system_changes",
                         "missing": list(missing_build)})
    camera = desired.get("virtual_camera", {})
    if (camera.get("desired") is True and not camera.get("load", True) and
            (not facts.get("v4l2", {}).get("module_loaded") or
             not facts.get("v4l2", {}).get("device_present"))):
        limitation = {"code": "v4l.load.disabled_device_unavailable", "applies_to": "virtual_camera"}
        if camera.get("required_for_success"):
            blockers.append(limitation)
        else:
            warnings.append(limitation)
    helper = facts.get("privileged_helper", {})
    helper_ready = bool(helper.get("trusted", False) and
                        helper.get("compatible", helper.get("trusted", False)) and
                        helper.get("authorization_transport_available", helper.get("trusted", False)))
    if any(op["privilege"] == "trusted_helper" for op in ops) and not helper_ready:
        if intent == "uninstall":
            ops = [op for op in ops if op["privilege"] != "trusted_helper"]
            previous: str | None = None
            for op in ops:
                op["depends_on"] = [] if previous is None else [previous]
                previous = op["id"]
            warnings.append({"code": "v4l.persistence.helper_unavailable_manual_cleanup",
                             "applies_to": "system_changes"})
        elif not any(item.get("applies_to") == "system_changes" and
                     item.get("code", "").startswith(("privileged_helper", "authorization_"))
                     for item in blockers):
            blockers.append({"code": helper.get("reason_codes", ["privileged_helper_missing"])[0],
                             "applies_to": "system_changes"})
    if verified_release:
        active_version = facts.get("installation", {}).get("active_version")
        try:
            update_decision = release_module().decide_update(
                active_version, verified_release["version"],
                allow_same_version=bool(getattr(args, "allow_same_version", False)),
                allow_downgrade=bool(getattr(args, "allow_downgrade", False)))
            if update_decision.action == "keep_current" and update_decision.relation in {"same", "installed_newer"}:
                blockers.append({"code": update_decision.reason_code, "applies_to": "version",
                                 "relation": update_decision.relation})
        except Exception:
            blockers.append({"code": "release.version.invalid", "applies_to": "version"})
        try:
            meets_minimum = release_module().installer_meets_minimum(
                installer_version(), verified_release["minimum_installer_version"])
        except Exception:
            meets_minimum = False
        if not meets_minimum:
            blockers.append({"code": "release.installer_update_required", "applies_to": "installer",
                             "minimum_version": verified_release["minimum_installer_version"],
                             "current_version": installer_version()})
    if (rec.get("selection_route", rec["route"]) == "recommended" and
            intent != "uninstall" and not verified_release):
        # A caller-supplied boolean is not provenance.  The release-channel
        # verifier must replace this blocker with signed manifest/archive
        # evidence when the two components are integrated.
        blockers.append({"code": "source.signed_verification_receipt_required", "applies_to": "source"})
    if getattr(args, "allow_unsupported", False) and intent != "uninstall":
        if rec.get("selection_route", rec["route"]) != "advanced":
            blockers.append({"code": "os.unsupported.override_requires_advanced",
                             "applies_to": "installation"})
        # Developer override permits only unprivileged work; automatic package or
        # module operations remain removed/fail-closed on unsupported systems.
        elif not facts.get("os", {}).get("supported", False):
            blockers = [b for b in blockers if not b["code"].startswith("os.unsupported")]
            ops = [op for op in ops if op["privilege"] != "trusted_helper"]
            previous = None
            for op in ops:
                op["depends_on"] = [] if previous is None else [previous]
                previous = op["id"]
            blockers = [b for b in blockers if b.get("code") != "privileged_helper_missing"]
            warnings.append({"code": "os.unsupported.developer_override_unprivileged_only", "applies_to": "installation"})
    if (desired.get("service", {}).get("desired") == "enabled_started" and
            not facts.get("systemd_user", {}).get("manager_usable", False)):
        warnings.append({"code": "service.user_manager.unavailable_degraded",
                         "applies_to": "service_changes"})
    created = utc_now()
    expires = (parse_time(created) + dt.timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    def reviewed_model_url(base: str | None, relative: str) -> str | None:
        if not base:
            return None
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme not in {"https", "http", "file"}:
            return None
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in relative.split("/"))
        return base.rstrip("/") + "/" + encoded

    downloads = ([{"artifact_id": "studiocast-source", "url": verified_release["archive_url"],
        "size": verified_release["archive_size"], "sha256": verified_release["archive_sha256"],
        "signature": verified_release["archive_signature"], "provenance": verified_release["provenance"],
        "license": verified_release["license"], "cache_path": verified_release["archive_path"],
        "destination": str(build_cache), "required_for_core": True}] if verified_release else []) + [
        {"artifact_id": f"{pack_id}:{artifact['name']}", "pack_id": pack_id,
         "filename": artifact["name"],
         "url": reviewed_model_url(model_source, artifact["source_paths"][0]),
         "source_base": model_source,
         "source_paths": copy.deepcopy(artifact["source_paths"]),
         "size": artifact["size_bytes"], "size_status": artifact["size_status"],
         "sha256": "sha256:" + artifact["sha256"], "signature": None,
         "provenance": copy.deepcopy(model_packs[pack_id]["provenance"]),
         "license": copy.deepcopy(model_packs[pack_id]["license"]),
         "cache_path": str(r["cache"] / "models/sha256" / artifact["sha256"]),
         "destination": str(model_destination / model_packs[pack_id]["family"] /
                            model_packs[pack_id]["pack_path"]),
         "required_for_core": False}
        for pack_id in ([] if intent == "uninstall" else desired.get("model_pack_ids", []))
        for artifact in model_packs[pack_id]["artifacts"]]
    base_required = sum(facts.get("paths", {}).get("required_bytes", {}).values())
    download_bytes = sum(item["size"] for item in downloads if type(item.get("size")) is int)
    required_bytes = base_required + download_bytes
    free_bytes = facts.get("paths", {}).get("free_bytes", 0)
    blockers = [item for item in blockers if item.get("code") != "disk.staging.insufficient"]
    if intent != "uninstall" and free_bytes < required_bytes:
        blockers.append({"code": "disk.staging.insufficient", "applies_to": "paths",
                         "free_bytes": free_bytes, "required_bytes": required_bytes})

    plan = {"schema_version": 1, "plan_version": PLAN_VERSION, "policy_version": POLICY_VERSION,
        "plan_id": plan_id, "created_at": created, "expires_at": expires, "intent": intent,
        "route": rec.get("selection_route", rec["route"]),
        "detected_route": rec["route"], "facts_fingerprint": facts["host_fingerprint"],
        "core_reuse": minimal_repair,
        "privilege_preconditions": {"base_os": facts.get("os", {}).get("base_id"),
            "base_release": facts.get("os", {}).get("base_version_id"),
            "kernel_release": facts.get("os", {}).get("kernel_release"),
            "helper": {key: copy.deepcopy(helper.get(key)) for key in (
                "path", "trusted", "compatible", "authorization_transport_available",
                "helper_version", "protocol_versions", "pkexec_path", "policy_path")}},
        "observed_state": {"service": copy.deepcopy(facts.get("service", {})),
                           "systemd_user": copy.deepcopy(facts.get("systemd_user", {})),
                           "v4l2": copy.deepcopy(facts.get("v4l2", {}))},
        "current_state": copy.deepcopy(facts.get("installation", {})), "desired_state": desired,
        "disk": {"free_bytes": free_bytes, "base_required_bytes": base_required,
                 "download_bytes": download_bytes, "required_bytes": required_bytes},
        "paths": {"build_cache": str(build_cache), "staging_payload": str(staging),
            "target_payload": str(target_payload), "current_link": str(current)},
        "source": ({"kind": "official_release", "path": source_path, "official": True,
                    "channel": verified_release["channel"], "version": verified_release["version"],
                    "manifest_sha256": verified_release["manifest_sha256"],
                    "manifest_key_id": verified_release["manifest_key_id"],
                    "archive_sha256": verified_release["archive_sha256"],
                    "archive_size": verified_release["archive_size"],
                    "archive_signature": verified_release["archive_signature"],
                    "provenance": verified_release["provenance"]}
                   if verified_release else
                   {"kind": source_kind, "path": source_path, "official": False,
                    "official_claimed": source_kind == "archive" and bool(getattr(args, "official_source", False))}),
        "release": ({"minimum_installer_version": verified_release["minimum_installer_version"],
                     "release_page_url": verified_release["release_page_url"],
                     "installer_appimage": verified_release["installer_appimage"]}
                    if verified_release else None),
        "downloads": downloads,
        "operations": ops, "preservation": preservation, "blockers": blockers, "warnings": warnings,
        "expected_validation": [{"code": "payload.binaries.executable"}, {"code": "manifest.atomic"}],
        "approval_token": uuid.uuid4().hex + uuid.uuid4().hex}
    plan["plan_digest"] = plan_digest(plan)
    persist_pending(plan)
    return plan
