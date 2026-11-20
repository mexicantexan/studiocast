"""Stable command-line parsing and structured output."""

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
    POLICY_VERSION,
    RESULT_VERSION,
    atomic_write,
    installer_version,
    load_json,
    roots,
)
from .executor import Executor, validate_plan
from .facts import host_analyze, integrate_release_facts, normalize_fixture
from .manifest import migrate_v1
from .packaged import (
    release_cache_dir,
    release_keys,
    release_module,
    verify_release_evidence,
)
from .plan import create_plan, persist_pending, plan_digest
from .recommendation import recommend, route_for

def legacy_plan_text(plan: dict[str, Any]) -> dict[str, Any]:
    """Compatibility fields retained while GUI migrates to operation objects."""
    out = copy.deepcopy(plan)
    cmake = next((op["inputs"] for op in plan["operations"] if op["id"] == "build.configure"), None)
    commands: list[str] = []
    if cmake:
        flags = " ".join(f"-D{k}={'ON' if v else 'OFF'}" for k, v in sorted(cmake["cmake"].items()))
        commands.append(f"cmake -S {cmake['source_path']} -B {cmake['build_path']} -DCMAKE_BUILD_TYPE={cmake['build_type']} {flags}")
    out["workflow"] = plan["intent"]
    out["supported"] = not any(x["code"].startswith("os.unsupported") for x in plan["blockers"])
    out["summary"] = [f"Current installed version: {plan['current_state'].get('active_version') or 'not installed'}"]
    if plan["intent"] != "uninstall": out["summary"] += [f"Target StudioCast version: {Path(plan['paths']['target_payload']).name}", f"Build directory: {plan['paths']['build_cache']}"]
    out["changes"] = ["Enable CPU resize fallback in the daemon config by default", "Force Linux CMake configuration to keep Open Video/Open CUDA and Open Audio enabled with explicit ON/OFF values"]
    out["privileged_operations"] = [op["id"] for op in plan["operations"] if op["privilege"] == "trusted_helper"]
    out["preserved"] = ["user settings", "downloaded model packs"] if plan["intent"] == "uninstall" and plan["preservation"]["preserve_settings"] else []
    out["removed"] = [op["id"] for op in plan["operations"] if op["kind"] == "cleanup"]
    out["commands"] = commands
    # Compatibility review fields are part of the reviewed document and are
    # therefore covered by the digest as well; execution never discards them.
    out["plan_digest"] = plan_digest(out)
    persist_pending(out)
    return out


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--route", choices=("recommended", "custom", "advanced"))
    parser.add_argument("--facts", "--facts-file", dest="facts_file")
    parser.add_argument("--source-dir")
    parser.add_argument("--release-archive")
    parser.add_argument("--official-source", action="store_true")
    parser.add_argument("--release-manifest")
    parser.add_argument("--release-signature")
    parser.add_argument("--release-receipt")
    parser.add_argument("--release-receipt-out")
    parser.add_argument("--release-cache-dir")
    parser.add_argument("--trusted-key", action="append", default=[])
    parser.add_argument("--stable-url")
    parser.add_argument("--stable-signature-url")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-same-version", action="store_true")
    parser.add_argument("--allow-downgrade", action="store_true")
    parser.add_argument("--build-dir")
    parser.add_argument("--build-type", default=None)
    parser.add_argument("--target-version")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--input-fixture")
    parser.add_argument("--with-deps", dest="deps", action="store_true")
    parser.add_argument("--skip-deps", dest="deps", action="store_false")
    parser.add_argument("--v4l2loopback", dest="v4l2", action="store_true")
    parser.add_argument("--no-v4l2loopback", dest="v4l2", action="store_false")
    parser.add_argument("--v4l-device-number", type=int)
    parser.add_argument("--v4l-label")
    parser.add_argument("--v4l-exclusive-caps", dest="v4l_exclusive_caps", action="store_true")
    parser.add_argument("--no-v4l-exclusive-caps", dest="v4l_exclusive_caps", action="store_false")
    parser.add_argument("--load-loopback", dest="load_loopback", action="store_true")
    parser.add_argument("--no-load-loopback", dest="load_loopback", action="store_false")
    parser.add_argument("--persist-loopback", dest="persist_loopback", action="store_true")
    parser.add_argument("--no-persist-loopback", dest="persist_loopback", action="store_false")
    parser.add_argument("--service", dest="service", action="store_true")
    parser.add_argument("--no-service", dest="service", action="store_false")
    parser.add_argument("--models", dest="models", action="store_true")
    parser.add_argument("--no-models", dest="models", action="store_false")
    parser.add_argument("--model-id", dest="model_ids", action="append", default=[])
    parser.add_argument("--model-destination")
    parser.add_argument("--model-source")
    parser.add_argument("--open-backends", dest="open_backends", action="store_true")
    parser.add_argument("--no-open-backends", dest="open_backends", action="store_false")
    parser.add_argument("--open-cuda", dest="open_cuda", action="store_true")
    parser.add_argument("--no-open-cuda", dest="open_cuda", action="store_false")
    parser.add_argument("--open-audio", dest="open_audio", action="store_true")
    parser.add_argument("--no-open-audio", dest="open_audio", action="store_false")
    parser.add_argument("--open-vulkan", dest="open_vulkan", action="store_true")
    parser.add_argument("--no-open-vulkan", dest="open_vulkan", action="store_false")
    parser.add_argument("--vulkan-runtime", dest="vulkan_runtime", action="store_true")
    parser.add_argument("--no-vulkan-runtime", action="store_true")
    parser.add_argument("--mesa-vulkan", dest="mesa_vulkan", action="store_true")
    parser.add_argument("--shader-tools", dest="shader_tools", action="store_true")
    parser.add_argument("--fresh-build", dest="fresh_build", action="store_true")
    parser.add_argument("--no-fresh-build", dest="fresh_build", action="store_false")
    parser.add_argument("--preserve-user-data", dest="preserve_settings", action="store_true")
    parser.add_argument("--remove-user-data", dest="preserve_settings", action="store_false")
    parser.set_defaults(v4l2=None, service=None, models=None, open_backends=None, open_cuda=None,
                        open_audio=None, open_vulkan=None, v4l_exclusive_caps=None,
                        preserve_settings=None, fresh_build=False, deps=None,
                        load_loopback=None, persist_loopback=None)


def output(value: Any, human: str | None = None) -> None:
    if human is not None: print(human)
    else: print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def failed_result(error: BackendError, plan: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "result_version": RESULT_VERSION,
            "plan_id": plan.get("plan_id"), "plan_digest": plan.get("plan_digest"),
            "transaction_id": None, "transaction_state": "failed", "state": "failed",
            "core_committed": False, "failed_operation_id": None,
            "error": {"code": error.code, "message": error.message, "details": error.details},
            "retryable_operation_ids": [],
            "exit": {"kind": "backend_result", "code": 2, "signal": None}}


def load_facts(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "facts_file", None):
        facts = normalize_fixture(load_json(Path(args.facts_file)))
    elif getattr(args, "input_fixture", None):
        facts = normalize_fixture(load_json(Path(args.input_fixture)))
    else:
        facts = host_analyze(getattr(args, "target_version", None))
    return integrate_release_facts(facts, args)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="studiocast-installer-backend")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_p = sub.add_parser("analyze"); add_common(analyze_p)
    detect_p = sub.add_parser("detect-os"); add_common(detect_p)
    status_p = sub.add_parser("status"); add_common(status_p)
    recommend_p = sub.add_parser("recommend"); add_common(recommend_p)
    recommend_p.add_argument("--intent", default="auto"); recommend_p.add_argument("--prior-desired"); recommend_p.add_argument("--policy-version", default=POLICY_VERSION)
    for name in ("plan", "run"):
        p = sub.add_parser(name); p.add_argument("workflow", choices=("install", "update", "repair", "reinstall", "clean-install", "uninstall", "modify", "advanced"), nargs="?", default="install"); add_common(p)
    for name in ("install", "update", "repair", "reinstall", "clean-install", "uninstall"):
        p = sub.add_parser(name); add_common(p); p.set_defaults(workflow=name)
    apply_p = sub.add_parser("apply-plan", aliases=["execute-plan"]); apply_p.add_argument("--plan", required=True); apply_p.add_argument("--digest", required=True); apply_p.add_argument("--token", required=True); apply_p.add_argument("--facts"); apply_p.add_argument("--dry-run", action="store_true"); apply_p.add_argument("--json", action="store_true")
    manifest_p = sub.add_parser("manifest"); manifest_p.add_argument("--migrate-v1", action="store_true")
    verify_release_p = sub.add_parser("verify-release"); add_common(verify_release_p)
    release_status_p = sub.add_parser("release-status"); add_common(release_status_p)
    release_status_p.add_argument("--installed-version")
    release_status_p.add_argument("--prepare-self-update", action="store_true")
    release_status_p.add_argument("--appimage-path")
    args = parser.parse_args(argv)
    if args.command in {"verify-release", "release-status"}:
        evidence = verify_release_evidence(args)
        if evidence is None:
            raise BackendError("release.manifest.required", "Signed release metadata is required")
        if args.command == "verify-release":
            output(evidence); return 0
        rc = release_module()
        installed = args.installed_version
        if installed is None:
            try:
                installed = host_analyze(evidence["version"])["installation"]["active_version"]
            except BackendError:
                installed = None
        decision = rc.decide_update(installed, evidence["version"],
                                    allow_same_version=args.allow_same_version,
                                    allow_downgrade=args.allow_downgrade)
        release_status: dict[str, Any] = {
            "schema_version": 1, "release_status_version": "installer-release-status/v1",
            "channel": evidence["channel"], "available_version": evidence["version"],
            "installed_version": installed, "action": decision.action, "relation": decision.relation,
            "reason_code": decision.reason_code, "minimum_installer_version": evidence["minimum_installer_version"],
            "installer_meets_minimum": rc.installer_meets_minimum(installer_version(), evidence["minimum_installer_version"]),
            "offline": bool(args.offline), "verified_source_archive": evidence["archive_path"],
            "source_archive_state": evidence["archive_state"],
            "source_acquisition_required": (evidence["archive_state"] == "acquisition_pending" and
                                            evidence.get("archive_cache_state") != "cached_verified"),
            "rollback": rc.rollback_metadata(evidence["version"], installed, roots()["data"] / "payloads")}
        if args.prepare_self_update:
            manifest_view = {"release": {"version": evidence["version"]},
                             "artifacts": {"installer_appimage": evidence["installer_appimage"]}}
            offer = rc.prepare_self_update(manifest_view, installer_version(), release_cache_dir(args) / "installer",
                                           release_keys(args, rc), appimage_path=args.appimage_path,
                                           offline=args.offline)
            release_status["self_update"] = {
                "state": offer.state, "reason_code": offer.reason_code,
                "current_appimage": offer.current_appimage,
                "verified_appimage": offer.verified_appimage,
                "relaunch_command": list(offer.relaunch_command),
                "manual_download_url": offer.manual_download_url,
                "requires_confirmation": offer.state == "offer_restart",
                "running_artifact_replaced": False}
        output(release_status); return 0
    if args.command in {"analyze", "detect-os"}:
        facts = load_facts(args)
        output(facts if args.command == "analyze" else facts["os"]); return 0
    if args.command == "status":
        evidence = verify_release_evidence(args)
        facts = load_facts(args); route, action = route_for(facts)
        value = {"schema_version": 2, "installed": facts["installation"]["classification"] in {"healthy", "partial", "stale"},
            "classification": facts["installation"]["classification"], "installed_version": facts["installation"].get("active_version"),
            "target_version": facts["installation"].get("target_version"), "version_relation": facts["installation"].get("version_relation"),
            "route": route, "primary_action": action, "reason_codes": facts["installation"]["reason_codes"],
            "stable_release": None if evidence is None else {"verified": True, "channel": evidence["channel"],
                "version": evidence["version"], "minimum_installer_version": evidence["minimum_installer_version"],
                "source_archive": evidence["archive_path"]},
            "optional_components": {"onnxruntime_cuda": facts["runtime"]["onnxruntime"], "vulkan": {**facts["gpus"]["vulkan"], "runtime_packages": ["libvulkan1", "vulkan-tools"], "mesa_icd_package": "mesa-vulkan-drivers", "install_docs": "docs/open_source_video_models_install.md"}, "maxine_sdk": {**facts["maxine"], "install_docs": "docs/maxine_install.md"}}}
        output(value); return 0
    if args.command == "recommend":
        facts = load_facts(args); prior = load_json(Path(args.prior_desired)) if args.prior_desired else None
        output(recommend(facts, args.intent, prior, args.policy_version, args.route or "recommended")); return 0
    if args.command in {"plan", "run", "install", "update", "repair", "reinstall", "clean-install", "uninstall"}:
        verify_release_evidence(args)
        facts = load_facts(args); prior = facts.get("installation", {}).get("desired_configuration") if args.workflow in {"update", "repair", "reinstall", "clean-install", "modify"} else None
        unverified_source = bool(args.source_dir or
            (args.release_archive and not getattr(args, "verified_release", None)) or
            args.model_source)
        inferred_route = "advanced" if args.workflow == "advanced" or unverified_source else "recommended"
        route = args.route or inferred_route
        if route != "advanced" and unverified_source:
            raise BackendError("source.arbitrary_requires_advanced",
                               "Arbitrary source and model locations require Advanced mode")
        rec = recommend(facts, args.workflow, prior, route=route)
        plan = create_plan(args, facts, rec)
        if args.command == "plan": output(legacy_plan_text(plan)); return 0
        if args.dry_run:
            result, code = Executor(plan, dry_run=True).execute(); output(result); return code
        try:
            validate_plan(plan, plan["plan_digest"], plan["approval_token"], facts)
        except BackendError as exc:
            print(f"{exc.code}: {exc.message}", file=sys.stderr)
            output(failed_result(exc, plan)); return 2
        try:
            result, code = Executor(plan).execute()
        except Exception as unexpected:
            error = (unexpected if isinstance(unexpected, BackendError) else
                     BackendError("operation.unexpected_failure", "Installer transaction could not start",
                                  {"exception_type": type(unexpected).__name__}))
            output(failed_result(error, plan)); return 2
        output(result); return code
    if args.command in {"apply-plan", "execute-plan"}:
        plan = load_json(Path(args.plan)); facts = normalize_fixture(load_json(Path(args.facts))) if args.facts else None
        try:
            validate_plan(plan, args.digest, args.token, facts)
        except BackendError as exc:
            print(f"{exc.code}: {exc.message}", file=sys.stderr)
            output(failed_result(exc, plan)); return 2
        try:
            result, code = Executor(plan, args.dry_run).execute()
        except Exception as unexpected:
            error = (unexpected if isinstance(unexpected, BackendError) else
                     BackendError("operation.unexpected_failure", "Installer transaction could not start",
                                  {"exception_type": type(unexpected).__name__}))
            output(failed_result(error, plan)); return 2
        output(result); return code
    if args.command == "manifest":
        path = roots()["data"] / "install-manifest.json"; value = load_json(path)
        if args.migrate_v1 and value.get("schema_version") == 1:
            value = migrate_v1(value, roots()); atomic_write(path, value)
        output(value); return 0
    return 2
