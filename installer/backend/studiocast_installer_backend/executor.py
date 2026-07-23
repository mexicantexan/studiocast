"""Validated transaction execution and rollback handling."""

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
    ALLOWED_OPERATION_KINDS,
    BINARIES,
    BackendError,
    DEFAULT_MODEL_PACKS,
    FORBIDDEN_HELPER_FIELDS,
    HELPER_PATH,
    PLAN_VERSION,
    POLICY_VERSION,
    PRIVILEGED_INPUT_FIELDS,
    PRIVILEGED_TYPES,
    PROGRESS_VERSION,
    RESULT_VERSION,
    SERVICE_NAME,
    V4L_OWNED_PATHS,
    atomic_write,
    canonical_bytes,
    file_sha256,
    is_beneath,
    load_json,
    parse_time,
    reject_duplicates,
    roots,
    safe_remove,
    service_unit_content,
    utc_now,
    validate_mutation_path,
)
from .facts import helper_facts, os_facts
from .manifest import manifest_state, migrate_v1
from .packaged import load_model_catalog, release_keys, release_module
from .plan import pending_path, plan_digest

class Executor:
    def __init__(self, plan: dict[str, Any], dry_run: bool = False):
        self.plan, self.dry_run, self.r = plan, dry_run, roots()
        self.manifest_path = self.r["data"] / "install-manifest.json"
        self.transaction_id = str(uuid.uuid4())
        self.journal: list[dict[str, Any]] = []
        self.executed: list[str] = []
        self.active_before: dict[str, Any] | None = None
        self.manifest_before: dict[str, Any] | None = None
        self.source_root = Path(plan["source"]["path"])
        self.reused_core = bool(plan.get("core_reuse"))
        self.core_committed = self.reused_core
        self.activated_new_payload = False
        self.rollback_state: str | None = None
        self.failed_operation: str | None = None
        self.retryable: list[str] = []
        self.model_results: dict[str, dict[str, Any]] = {}
        self.progress_sequence = 0
        self.cancelled = False
        self.child: subprocess.Popen[bytes] | None = None
        self.settings_snapshot: Path | None = None
        observed = plan.get("observed_state", {})
        self.service_actual = observed.get("service", {}).get("actual", "unknown")
        self.v4l_actual = copy.deepcopy(observed.get("v4l2", {}))
        self.v4l_owned_configuration = copy.deepcopy(
            (self.manifest_before or {}).get("v4l", {}).get("owned_configuration", []))
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists(): return
        value = load_json(self.manifest_path)
        if value.get("schema_version") == 1:
            value = migrate_v1(value, self.r)
        elif value.get("schema_version") != 2:
            raise BackendError("manifest.unknown_schema", "Cannot execute with unknown manifest schema")
        self.manifest_before = value
        self.active_before = copy.deepcopy(value.get("payloads", {}).get("active"))
        self.service_actual = value.get("service", {}).get("actual", self.service_actual)
        self.v4l_owned_configuration = copy.deepcopy(
            value.get("v4l", {}).get("owned_configuration", self.v4l_owned_configuration))

    def run_process(self, argv: list[str]) -> None:
        if self.dry_run:
            print("+ " + " ".join(argv))
            return
        self.child = subprocess.Popen(argv, start_new_session=True)
        previous_handlers: dict[int, Any] = {}
        def cancel(signum: int, _frame: Any) -> None:
            self.cancelled = True
            raise BackendError("operation.cancelled", f"Operation cancelled by signal {signum}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, cancel)
        try:
            code = self.child.wait()
        except (KeyboardInterrupt, BackendError) as exc:
            self.cancelled = True
            try: os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                self.child.wait()
            if isinstance(exc, BackendError):
                raise
            raise BackendError("operation.cancelled", "Operation cancelled") from exc
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self.child = None
        if code != 0:
            raise BackendError("process.exit", f"Process exited with status {code}", {"argv0": argv[0], "exit_code": code})

    def append(self, op_id: str, state: str, error: BackendError | None = None) -> None:
        self.journal.append({"sequence": len(self.journal) + 1, "operation_id": op_id, "state": state,
            "finished_at": utc_now(), "exit": {"kind": "backend_result", "code": 0 if error is None else 1, "signal": None},
            "error": None if error is None else {"code": error.code, "message": error.message, "details": error.details}, "compensation_operation_ids": []})
        if not self.dry_run:
            checkpoint = "cancelled" if self.cancelled else "core_committed" if self.core_committed else "executing"
            self.write_manifest(checkpoint, error)

    def progress(self, op: dict[str, Any], boundary: str, state: str,
                 error: BackendError | None = None) -> None:
        self.progress_sequence += 1
        event = {"schema_version": 1, "event_version": PROGRESS_VERSION,
            "progress_version": PROGRESS_VERSION,
            "plan_id": self.plan["plan_id"], "plan_digest": self.plan["plan_digest"],
            "transaction_id": self.transaction_id, "sequence": self.progress_sequence,
            "operation_id": op["id"], "phase": op.get("review", {}).get("category", "operation"),
            "boundary": boundary, "state": state, "timestamp": utc_now(),
            "error": None if error is None else {"code": error.code, "message": error.message,
                                                   "details": error.details}}
        print(json.dumps(event, sort_keys=True, ensure_ascii=False), flush=True)

    def execute(self) -> tuple[dict[str, Any], int]:
        completed: set[str] = set()
        failure: BackendError | None = None
        for op in self.plan["operations"]:
            if any(dep not in completed for dep in op["depends_on"]):
                failure = BackendError("plan.dependency_order", f"Dependencies not completed for {op['id']}")
                self.failed_operation = op["id"]; break
            self.executed.append(op["id"])
            self.progress(op, "before", "started")
            try:
                self.execute_one(op)
                completed.add(op["id"])
                self.append(op["id"], "completed")
                self.progress(op, "after", "completed")
            except Exception as unexpected:
                exc = (unexpected if isinstance(unexpected, BackendError) else
                       BackendError("operation.unexpected_failure",
                                    f"Unexpected failure while executing {op['id']}",
                                    {"exception_type": type(unexpected).__name__}))
                self.append(op["id"], "cancelled" if self.cancelled else "failed", exc)
                self.progress(op, "after", "cancelled" if self.cancelled else "failed", exc)
                self.failed_operation = op["id"]
                if op["failure_policy"] == "degrade" and (self.core_committed or self.plan["intent"] == "uninstall"):
                    self.retryable.append(op["id"])
                    # Preserve exact-plan accounting: remaining optional model
                    # operations are recorded skipped, then manifest commits.
                    completed.add(op["id"])
                    continue
                failure = exc
                break
        state = "cancelled" if self.cancelled else "failed" if failure else "degraded" if self.retryable else "committed"
        if state in {"failed", "cancelled"} and self.activated_new_payload:
            self.rollback_after_activation(failure)
        if state in {"failed", "cancelled"}:
            self.compensate_settings()
        if self.rollback_state == "rolled_back":
            state = "rolled_back"
        if not self.dry_run:
            self.write_manifest(state, failure)
        result = {"schema_version": 1, "result_version": RESULT_VERSION,
            "plan_id": self.plan["plan_id"], "plan_digest": self.plan["plan_digest"],
            "transaction_id": self.transaction_id, "transaction_state": state, "state": state,
            "rollback_state": self.rollback_state,
            "core_committed": self.core_committed, "active_version": self.plan["desired_state"].get("version") or self.plan["paths"]["target_payload"].rstrip("/").split("/")[-1] if self.core_committed else (self.active_before or {}).get("version"),
            "failed_operation_id": self.failed_operation, "error": None if failure is None else {"code": failure.code, "message": failure.message, "details": failure.details},
            "retryable_operation_ids": self.retryable, "reviewed_operation_ids": [op["id"] for op in self.plan["operations"]],
            "executed_operation_ids": self.executed, "exit": {"kind": "backend_result", "code": 0 if state == "committed" else 3 if state == "degraded" else 130 if state == "cancelled" else 2, "signal": None}}
        return result, result["exit"]["code"]

    def execute_one(self, op: dict[str, Any]) -> None:
        kind, inputs = op["kind"], op["inputs"]
        if kind not in ALLOWED_OPERATION_KINDS:
            raise BackendError("plan.unknown_operation", f"Unknown operation kind: {kind}")
        if kind == "preflight":
            # Required link ownership is checked before payload activation so a
            # conflicting user file cannot produce a half-switched install.
            if any(item["kind"] == "links_reconcile" and item["inputs"].get("desired") == "present" for item in self.plan["operations"]):
                for name in BINARIES:
                    link = self.r["bin"] / name
                    if link.exists() and not link.is_symlink():
                        raise BackendError("ownership.mismatch", f"Refusing to replace non-link: {link}")
                    expected = self.r["data"] / "current/bin" / name
                    if link.is_symlink() and link.readlink() != expected:
                        raise BackendError("ownership.mismatch", f"Refusing to replace unowned link: {link}")
            return
        if kind == "settings_snapshot":
            src, dst = self.r["config"], Path(inputs["snapshot_path"])
            validate_mutation_path(dst, [self.r["config"].parent])
            self.settings_snapshot = dst
            if not self.dry_run and src.exists():
                if dst.exists(): safe_remove(dst, [self.r["config"].parent])
                shutil.copytree(src, dst, symlinks=True)
        elif kind == "release_acquire": self.acquire_release(inputs)
        elif kind == "source_extract": self.extract_source(Path(inputs["archive"]), Path(inputs["destination"]))
        elif kind == "build_cache_prepare":
            path = Path(self.plan["paths"]["build_cache"]); validate_mutation_path(path, [self.r["cache"]])
            if inputs.get("fresh") and path.exists() and not self.dry_run: safe_remove(path, [self.r["cache"]])
            if not self.dry_run: path.mkdir(parents=True, exist_ok=True)
        elif kind == "build_configure":
            source = self.source_root
            if inputs["source_kind"] == "archive": source = self.source_root
            cmake = inputs["cmake"]
            argv = ["cmake", "-S", str(source), "-B", inputs["build_path"], "-G", "Ninja", f"-DCMAKE_BUILD_TYPE={inputs['build_type']}"]
            argv += [f"-D{name}={'ON' if bool(value) else 'OFF'}" for name, value in sorted(cmake.items())]
            self.run_process(argv)
        elif kind == "build_compile": self.run_process(["cmake", "--build", inputs["build_path"], "--target", *inputs["targets"]])
        elif kind == "payload_stage": self.stage_payload(Path(inputs["build_path"]))
        elif kind == "payload_validate": self.validate_staging(inputs["required_binaries"])
        elif kind == "trusted_helper": self.run_helper(inputs)
        elif kind == "payload_activate": self.activate_payload(inputs["version"])
        elif kind == "links_reconcile": self.reconcile_links(inputs.get("desired", "present"), inputs.get("binaries", BINARIES))
        elif kind == "service_unit_reconcile": self.reconcile_service_unit(inputs.get("desired", "disabled"))
        elif kind == "service_state_reconcile": self.reconcile_service_state(inputs.get("desired", "disabled"))
        elif kind == "model_transaction": self.install_model_pack(inputs)
        elif kind == "settings_restore": self.restore_settings(Path(inputs["snapshot_path"]))
        elif kind == "cleanup": self.cleanup(inputs["target"])
        elif kind in {"manifest_commit", "uninstall_tombstone"}: return

    def acquire_release(self, inputs: dict[str, Any]) -> None:
        rc = release_module()
        keys = release_keys(argparse.Namespace(trusted_key=[]), rc)
        manifest_path = Path(inputs["manifest_path"])
        signature_path = Path(inputs["signature_path"])
        try:
            raw_manifest = manifest_path.read_bytes()
            raw_signature = signature_path.read_bytes()
            if hashlib.sha256(raw_manifest).hexdigest() != inputs["manifest_sha256"]:
                raise BackendError("release.manifest.changed", "Signed release manifest changed after review")
            manifest = rc.verify_manifest(raw_manifest, raw_signature, keys,
                allow_file_urls=os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1")
            envelope = rc.strict_json_loads(raw_signature)
        except BackendError:
            raise
        except (OSError, rc.ReleaseChannelError) as exc:
            raise BackendError(getattr(exc, "code", "release.manifest.unavailable"), str(exc)) from exc
        if (manifest["release"]["version"] != inputs["version"] or
                envelope.get("key_id") != inputs["manifest_key_id"] or
                canonical_bytes(manifest["artifacts"]["source_archive"]) !=
                canonical_bytes(inputs["artifact"])):
            raise BackendError("release.evidence_changed", "Release evidence differs from the reviewed plan")
        metadata = inputs["artifact"]
        cache_dir = Path(inputs["cache_dir"])
        target = cache_dir / metadata["filename"]
        if self.dry_run:
            self.source_root = target
            return
        candidate_value = inputs.get("candidate_path")
        try:
            target_good = False
            if target.exists() and not target.is_symlink():
                try:
                    rc.verify_artifact(target, metadata, keys)
                    target_good = True
                except rc.ReleaseChannelError:
                    target_good = False
            if not target_good and candidate_value:
                candidate = Path(candidate_value)
                rc.verify_artifact(candidate, metadata, keys)
                cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                if cache_dir.is_symlink():
                    raise BackendError("release.cache.unsafe_path", "Release cache may not be a symlink")
                temporary = cache_dir / f".{metadata['filename']}.{uuid.uuid4().hex}.tmp"
                try:
                    shutil.copy2(candidate, temporary, follow_symlinks=False)
                    rc.verify_artifact(temporary, metadata, keys)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists(): temporary.unlink()
            elif not target_good:
                target = rc.acquire_artifact(metadata, cache_dir, keys,
                    offline=bool(inputs["offline"]))
            rc.verify_artifact(target, metadata, keys)
        except BackendError:
            raise
        except (OSError, rc.ReleaseChannelError) as exc:
            raise BackendError(getattr(exc, "code", "release.artifact.acquire_failed"), str(exc)) from exc
        self.source_root = target

    def extract_source(self, archive: Path, destination: Path) -> None:
        validate_mutation_path(destination, [self.r["cache"]])
        if self.dry_run: self.source_root = destination; return
        destination.mkdir(parents=True, exist_ok=False)
        def safe_name(name: str) -> bool:
            p = Path(name)
            return not p.is_absolute() and ".." not in p.parts
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                if any(not safe_name(m.name) or m.issym() or m.islnk() for m in tf.getmembers()):
                    raise BackendError("archive.unsafe_entry", "Archive contains an unsafe path or link")
                tf.extractall(destination, filter="data")
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                if any(not safe_name(info.filename) for info in zf.infolist()):
                    raise BackendError("archive.unsafe_entry", "Archive contains an unsafe path")
                zf.extractall(destination)
        else: raise BackendError("archive.unsupported", f"Unsupported archive: {archive}")
        candidates = [destination] + [p.parent for p in destination.rglob("VERSION") if (p.parent / "CMakeLists.txt").is_file()]
        self.source_root = next((p for p in candidates if (p / "CMakeLists.txt").is_file() and (p / "VERSION").is_file()), destination)

    def stage_payload(self, build: Path) -> None:
        staging = Path(self.plan["paths"]["staging_payload"]); validate_mutation_path(staging, [self.r["data"]])
        if self.dry_run: return
        if staging.exists(): safe_remove(staging, [self.r["data"]])
        (staging / "bin").mkdir(parents=True)
        for name in BINARIES:
            source = build / name
            if not source.is_file() or source.is_symlink(): raise BackendError("payload.binary_missing", f"Built binary missing: {source}")
            shutil.copy2(source, staging / "bin" / name, follow_symlinks=False)

    def validate_staging(self, names: list[str]) -> None:
        if self.dry_run: return
        staging = Path(self.plan["paths"]["staging_payload"])
        for name in names:
            path = staging / "bin" / name
            if not path.is_file() or not os.access(path, os.X_OK): raise BackendError("payload.validation_failed", f"Payload binary not executable: {path}")
        expected = next((op["inputs"].get("version") for op in self.plan["operations"] if op["id"] == "payload.stage"), None)
        try:
            smoke = subprocess.run([str(staging / "bin/studiocast"), "--version"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError("payload.smoke_test_failed", "Staged StudioCast version probe could not run") from exc
        if smoke.returncode != 0:
            raise BackendError("payload.smoke_test_failed", "Staged StudioCast version probe failed",
                               {"exit_code": smoke.returncode})
        reported = (smoke.stdout + " " + smoke.stderr).strip()
        if expected and expected != "unknown" and expected not in reported:
            raise BackendError("payload.version_mismatch", "Staged payload reported a different version",
                               {"expected": expected, "reported": reported})

    def run_helper(self, inputs: dict[str, Any]) -> None:
        if any(key in FORBIDDEN_HELPER_FIELDS for key in inputs): raise BackendError("privilege.forbidden_field", "Privileged request contains a forbidden field")
        if inputs.get("operation_type") not in PRIVILEGED_TYPES: raise BackendError("privilege.unknown_operation", "Unknown privileged operation")
        test_mode = os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1"
        helper_path = HELPER_PATH
        pkexec = Path("/usr/bin/pkexec")
        if test_mode:
            helper_path = Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_HELPER", str(helper_path))).absolute()
            pkexec = Path(os.environ.get("STUDIOCAST_INSTALLER_TEST_PKEXEC", str(pkexec))).absolute()
            if not helper_path.is_file() or not pkexec.is_file():
                raise BackendError("privileged_helper_missing", "Test privilege adapters are missing")
        else:
            facts = helper_facts()
            if not (facts["trusted"] and facts["compatible"] and
                    facts["authorization_transport_available"]):
                raise BackendError(facts["reason_codes"][0],
                                   "Trusted StudioCast system helper authorization is unavailable")
        request = {"schema_version": 1, "request_id": str(uuid.uuid4()), "transaction_id": self.transaction_id,
            "policy_version": 1, "plan_digest": self.plan["plan_digest"],
            "preconditions": {key: self.plan["privilege_preconditions"][key]
                              for key in ("base_os", "base_release", "kernel_release")},
            "operations": [{"id": self.executed[-1], "type": inputs["operation_type"],
                            **{k: v for k, v in inputs.items() if k != "operation_type"}}]}
        if self.dry_run: return
        if not pkexec.is_file():
            raise BackendError("authorization_unavailable", "The fixed polkit authorization client is unavailable")
        try:
            completed = subprocess.run([str(pkexec), str(helper_path), "--json-stdin"], input=canonical_bytes(request),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=float(os.environ.get("STUDIOCAST_INSTALLER_TEST_AUTH_TIMEOUT", "120")) if test_mode else 120)
        except subprocess.TimeoutExpired as exc:
            raise BackendError("authorization_timeout", "Authorization timed out") from exc
        # pkexec has stable process outcomes; helper results remain structured on
        # stdout.  There is deliberately no sudo/password fallback.
        if completed.returncode == 126:
            raise BackendError("authorization_denied", "Authorization was denied")
        if completed.returncode == 127:
            raise BackendError("authorization_cancelled", "Authorization was cancelled or unavailable")
        if completed.returncode:
            raise BackendError("privilege.operation_failed", "Trusted helper operation failed", {"exit_code": completed.returncode})
        try: response = json.loads(completed.stdout, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as exc: raise BackendError("privilege.malformed_result", "Trusted helper returned invalid JSON") from exc
        except BackendError as exc: raise BackendError("privilege.malformed_result", "Trusted helper returned duplicate JSON fields") from exc
        if not isinstance(response, dict):
            raise BackendError("privilege.malformed_result", "Trusted helper result must be an object")
        if (response.get("schema_version") != 1 or response.get("request_id") != request["request_id"] or
                response.get("transaction_id") != request["transaction_id"] or
                response.get("plan_digest") != request["plan_digest"]):
            raise BackendError("privilege.result_binding_mismatch", "Trusted helper result is not bound to this request")
        results = response.get("results")
        if (not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict) or
                results[0].get("id") != self.executed[-1] or results[0].get("type") != inputs["operation_type"]):
            raise BackendError("privilege.result_binding_mismatch", "Trusted helper operation result does not match the reviewed operation")
        if response.get("state") != "succeeded" or results[0].get("status") != "succeeded":
            raise BackendError(response.get("reason_code", "privilege.operation_failed"), "Trusted helper did not succeed")
        if inputs["operation_type"] == "v4l.persistence.write.v1":
            files = results[0].get("files", [])
            validated: list[dict[str, Any]] = []
            for item in files:
                if (isinstance(item, dict) and item.get("id") in V4L_OWNED_PATHS and
                        item.get("path") == str(V4L_OWNED_PATHS[item["id"]]) and
                        re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("sha256", "")))):
                    validated.append({"id": item["id"], "path": item["path"], "sha256": item["sha256"]})
            if len(validated) != len(V4L_OWNED_PATHS):
                raise BackendError("privilege.malformed_result", "Persistence result omitted owned-file hashes")
            self.v4l_owned_configuration = validated
            self.v4l_actual["persistence_state"] = "configured"
        elif inputs["operation_type"] == "v4l.persistence.remove.v1":
            self.v4l_owned_configuration = []
            self.v4l_actual["persistence_state"] = "absent"
        elif inputs["operation_type"] == "v4l.module.load.v1":
            self.v4l_actual.update({"module_available": True, "module_loaded": True,
                                    "device_present": True, "device_number": inputs["device_number"],
                                    "device_label": inputs["label"]})

    def activate_payload(self, version: str) -> None:
        staging, target, current = map(Path, (self.plan["paths"]["staging_payload"], self.plan["paths"]["target_payload"], self.plan["paths"]["current_link"]))
        for path in (staging, target, current): validate_mutation_path(path, [self.r["data"]])
        if self.dry_run: self.core_committed = True; return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): safe_remove(target, [self.r["data"]])
        os.replace(staging, target)
        tmp_link = current.with_name(f".{current.name}.{uuid.uuid4().hex}.tmp")
        os.symlink(str(target), tmp_link)
        os.replace(tmp_link, current)
        self.core_committed = True
        self.activated_new_payload = True

    def rollback_after_activation(self, failure: BackendError | None) -> None:
        self.rollback_state = "rolling_back"
        self.journal.append({"sequence": len(self.journal) + 1,
            "operation_id": "payload.rollback", "state": "rolling_back", "finished_at": utc_now(),
            "exit": {"kind": "backend_result", "code": 0, "signal": None}, "error": None,
            "compensation_operation_ids": ["payload.activate"]})
        if not self.dry_run:
            self.write_manifest("rolling_back", failure)
        try:
            current = self.r["data"] / "current"
            validate_mutation_path(current, [self.r["data"]])
            expected_target = Path(self.plan["paths"]["target_payload"])
            if not current.is_symlink() or current.readlink() != expected_target:
                raise BackendError("rollback.current_pointer_changed",
                                   "The active pointer changed during rollback")
            if self.active_before:
                previous = Path(self.active_before.get("path", ""))
                if (not is_beneath(previous, self.r["data"] / "payloads") or
                        not previous.is_dir() or previous.is_symlink()):
                    raise BackendError("rollback.previous_payload_unavailable",
                                       "The previous durable payload is unavailable")
                if not self.dry_run:
                    tmp = current.with_name(f".{current.name}.rollback-{uuid.uuid4().hex}")
                    try:
                        os.symlink(previous, tmp)
                        os.replace(tmp, current)
                    finally:
                        if tmp.is_symlink():
                            tmp.unlink()
            elif not self.dry_run:
                current.unlink()
            self.core_committed = False
            self.rollback_state = "rolled_back"
            rollback_error = None
        except (OSError, BackendError) as exc:
            rollback_error = (exc if isinstance(exc, BackendError) else
                              BackendError("rollback.pointer_restore_failed", str(exc)))
            self.rollback_state = "rollback_failed"
        self.journal.append({"sequence": len(self.journal) + 1,
            "operation_id": "payload.rollback", "state": self.rollback_state,
            "finished_at": utc_now(),
            "exit": {"kind": "backend_result", "code": 0 if rollback_error is None else 1,
                     "signal": None},
            "error": None if rollback_error is None else {"code": rollback_error.code,
                "message": rollback_error.message, "details": rollback_error.details},
            "compensation_operation_ids": ["payload.activate"]})

    def reconcile_links(self, desired: str, names: Any) -> None:
        self.r["bin"].mkdir(parents=True, exist_ok=True)
        for name in names:
            link = self.r["bin"] / name; validate_mutation_path(link, [self.r["bin"]])
            expected = self.r["data"] / "current/bin" / name
            if desired == "absent":
                if link.is_symlink() and link.readlink() == expected: link.unlink()
                continue
            if self.dry_run: continue
            target = self.r["data"] / "current/bin" / name
            if link.exists() and not link.is_symlink(): raise BackendError("ownership.mismatch", f"Refusing to replace non-link: {link}")
            temp = link.with_name(f".{name}.{uuid.uuid4().hex}.tmp"); os.symlink(target, temp); os.replace(temp, link)

    def reconcile_service_unit(self, desired: str) -> None:
        unit = self.r["systemd"] / SERVICE_NAME; validate_mutation_path(unit, [self.r["systemd"]])
        canonical = service_unit_content(self.r["data"])
        recorded = (self.manifest_before or {}).get("service", {})
        recorded_hash = (recorded.get("unit_sha256") if recorded.get("unit_path") == str(unit) else None)
        owned = False
        if unit.is_file() and not unit.is_symlink():
            current_hash = file_sha256(unit)
            owned = (unit.read_text(encoding="utf-8", errors="strict") == canonical or
                     (re.fullmatch(r"sha256:[0-9a-f]{64}", str(recorded_hash or "")) is not None and
                      current_hash == recorded_hash))
        if desired in {"absent", "disabled"}:
            if not self.dry_run and owned: unit.unlink()
            return
        if self.dry_run: return
        if (unit.exists() or unit.is_symlink()) and not owned:
            raise BackendError("ownership.mismatch", "Refusing to replace an unowned StudioCast service unit")
        unit.parent.mkdir(parents=True, exist_ok=True)
        temporary = unit.with_name(f".{unit.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(canonical, encoding="utf-8")
            temporary.chmod(0o644)
            os.replace(temporary, unit)
        finally:
            if temporary.exists(): temporary.unlink()

    def reconcile_service_state(self, desired: str) -> None:
        manager = self.plan.get("observed_state", {}).get("systemd_user", {})
        if desired == "enabled_started" and not manager.get("manager_usable", False):
            raise BackendError("service.user_manager.unavailable",
                               "The reviewed host has no usable user systemd manager")
        if self.dry_run: return
        if os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1":
            adapter = os.environ.get("STUDIOCAST_INSTALLER_TEST_SYSTEMCTL")
            if not adapter:
                if desired == "enabled_started":
                    raise BackendError("service.test_adapter_missing",
                                       "Hermetic service adapter is unavailable")
                self.service_actual = "inactive"
                return
        systemctl = (os.environ.get("STUDIOCAST_INSTALLER_TEST_SYSTEMCTL")
                     if os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1" else
                     shutil.which("systemctl"))
        if not systemctl:
            if desired == "enabled_started":
                raise BackendError("service.systemctl.unavailable", "systemctl is unavailable")
            self.service_actual = "inactive"
            return
        if desired == "enabled_started":
            self.run_process([systemctl, "--user", "daemon-reload"])
            self.run_process([systemctl, "--user", "enable", "--now", SERVICE_NAME])
            self.service_actual = "active"
        else:
            completed = subprocess.run([systemctl, "--user", "disable", "--now", SERVICE_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if completed.returncode not in {0, 1, 5}: raise BackendError("service.state_failed", "Could not disable StudioCast service")
            self.service_actual = "inactive"

    def install_model_pack(self, inputs: dict[str, Any]) -> None:
        module, catalog, component_root, catalog_hash = load_model_catalog()
        current = next((pack for pack in catalog["packs"] if pack["id"] == inputs["pack_id"]), None)
        if (catalog_hash != inputs["catalog_sha256"] or
                catalog["catalog_version"] != inputs["catalog_version"] or
                current is None or canonical_bytes(current) != canonical_bytes(inputs["pack"]) or
                current["family"] != inputs["family"]):
            raise BackendError("model.catalog.changed", "Packaged model catalog differs from the reviewed plan")
        if self.dry_run:
            return
        try:
            transport = None if inputs["offline"] else module.transport_for(inputs["source"])
            installer = module.ModelTransactionInstaller(catalog, component_root,
                Path(inputs["destination_root"]) / inputs["family"], self.r["cache"] / "models",
                transport, offline=bool(inputs["offline"]), retries=int(inputs["retries"]))
            result = installer.install(inputs["family"], [inputs["pack_id"]], force=False,
                                       recommended=bool(inputs["recommended"]))
        except (OSError, module.CatalogError, module.TransactionError) as exc:
            raise BackendError(getattr(exc, "reason_code", "model.transaction_failed"), str(exc)) from exc
        self.model_results[inputs["pack_id"]] = copy.deepcopy(result)
        if result.get("state") != "succeeded":
            pack_result = (result.get("packs") or [{}])[0]
            raise BackendError(str(pack_result.get("reason_code") or result.get("reason_code") or
                                   "model.transaction_failed"),
                               str(pack_result.get("message") or "Curated model transaction failed"),
                               {"pack_id": inputs["pack_id"]})

    def restore_settings(self, snapshot: Path) -> None:
        if self.dry_run or not snapshot.exists(): return
        if self.r["config"].exists(): safe_remove(self.r["config"], [self.r["config"].parent])
        os.replace(snapshot, self.r["config"])
        self.settings_snapshot = None

    def compensate_settings(self) -> None:
        snapshot = self.settings_snapshot
        if self.dry_run or snapshot is None or not snapshot.exists():
            return
        error: BackendError | None = None
        try:
            validate_mutation_path(snapshot, [self.r["config"].parent])
            if self.r["config"].exists():
                # Failure before the reset left the authoritative settings in
                # place; do not overwrite a possibly newer user edit.
                safe_remove(snapshot, [self.r["config"].parent])
            else:
                os.replace(snapshot, self.r["config"])
        except (OSError, BackendError) as exc:
            error = exc if isinstance(exc, BackendError) else BackendError("settings.compensation_failed", str(exc))
        self.journal.append({"sequence": len(self.journal) + 1,
            "operation_id": "settings.restore.compensation",
            "state": "failed" if error else "completed", "finished_at": utc_now(),
            "exit": {"kind": "backend_result", "code": 1 if error else 0, "signal": None},
            "error": None if error is None else {"code": error.code, "message": error.message,
                                                   "details": error.details},
            "compensation_operation_ids": ["settings.snapshot"]})
        self.settings_snapshot = None

    def cleanup(self, target: str) -> None:
        mapping = {"payloads": self.r["data"] / "payloads",
            "runtime_state": self.r["state"], "models": self.r["data"] / "models", "settings": self.r["config"]}
        if target == "service":
            self.reconcile_service_state("disabled"); self.reconcile_service_unit("absent"); return
        if target == "data":
            # Ordinary uninstall removes fixed installer metadata, never arbitrary
            # paths read from a stale manifest.
            for path in (self.r["data"] / "current", self.r["data"] / "install-manifest.json"):
                if not self.dry_run: safe_remove(path, [self.r["data"]])
            return
        if target == "build_cache":
            for path in (self.r["cache"] / "builds", self.r["cache"] / "sources"):
                if path.exists() and not self.dry_run: safe_remove(path, [self.r["cache"]])
            return
        if target == "cache":
            if self.r["cache"].exists() and not self.dry_run:
                safe_remove(self.r["cache"], [self.r["cache"].parent])
            return
        path = mapping.get(target)
        if path and path.exists() and not self.dry_run: safe_remove(path, [path.parent])

    def write_manifest(self, state: str, failure: BackendError | None) -> None:
        version = Path(self.plan["paths"]["target_payload"]).name
        target = Path(self.plan["paths"]["target_payload"])
        active = (self.active_before if self.reused_core else
                  {"version": version, "path": str(target), "tree_hash": None,
                   "validation": self.plan["expected_validation"]}) if self.core_committed else self.active_before
        previous = self.active_before if self.core_committed else (self.manifest_before or {}).get("payloads", {}).get("previous")
        if self.plan["intent"] == "uninstall": active = None
        unit_path = self.r["systemd"] / SERVICE_NAME
        unit_hash = file_sha256(unit_path) if unit_path.is_file() and not unit_path.is_symlink() else None
        reviewed_model_ops = [op for op in self.plan["operations"] if op["kind"] == "model_transaction"]
        previous_models = (self.manifest_before or {}).get("models", {}).get("packs", {})
        desired_model_ids = set(self.plan["desired_state"].get("model_pack_ids", []))
        model_manifest: dict[str, Any] = {
            pack_id: copy.deepcopy(value) for pack_id, value in previous_models.items()
            if pack_id in desired_model_ids and isinstance(value, dict)
        }
        for op in reviewed_model_ops:
            inputs = op["inputs"]
            pack = inputs["pack"]
            destination = Path(inputs["destination_root"]) / pack["family"] / pack["pack_path"]
            result = self.model_results.get(inputs["pack_id"])
            model_manifest[inputs["pack_id"]] = {
                "family": pack["family"], "task": pack["task"],
                "destination": str(destination),
                "catalog_version": inputs["catalog_version"],
                "catalog_sha256": inputs["catalog_sha256"],
                "license": copy.deepcopy(pack["license"]),
                "provenance": copy.deepcopy(pack["provenance"]),
                "artifacts": copy.deepcopy(pack["artifacts"]),
                "state": "installed" if destination.is_dir() and not destination.is_symlink() else "missing",
                "transaction_result": copy.deepcopy(result)}
        manifest = {"schema_version": 2, "policy_version": POLICY_VERSION,
            "transaction": {"id": self.transaction_id, "state": "uninstalled" if self.plan["intent"] == "uninstall" and state == "committed" else state,
                "intent": self.plan["intent"], "started_at": self.plan["created_at"], "updated_at": utc_now(), "completed_at": utc_now(),
                "plan_id": self.plan["plan_id"], "plan_digest": self.plan["plan_digest"]},
            "install_scope": "user", "versions": {"active": active.get("version") if active else None,
                "previous": previous.get("version") if previous else None, "target": version, "relation_at_start": self.plan["current_state"].get("version_relation")},
            "payloads": {"root": str(self.r["data"] / "payloads"), "current_link": str(self.r["data"] / "current"),
                "active": active, "previous": previous, "target": None},
            "source": {"channel": "stable" if self.plan["source"].get("official") else "advanced", "official": self.plan["source"].get("official", False),
                "archive_sha256": file_sha256(Path(self.plan["source"]["path"])) if self.plan["source"]["kind"] in {"archive", "official_release"} and Path(self.plan["source"]["path"]).is_file() else None},
            "desired_configuration": {"build_type": self.plan["desired_state"]["build_type"], "features": self.plan["desired_state"]["features"],
                "service": self.plan["desired_state"]["service"], "v4l": self.plan["desired_state"]["virtual_camera"],
                "model_pack_ids": self.plan["desired_state"].get("model_pack_ids", []),
                "model_destination": self.plan["desired_state"].get("model_destination"),
                "model_source": self.plan["desired_state"].get("model_source"),
                "model_offline": self.plan["desired_state"].get("model_offline", False),
                "preserve_settings": self.plan["preservation"]["preserve_settings"]},
            "effective_build_configuration": {"cmake": next((op["inputs"]["cmake"] for op in self.plan["operations"] if op["id"] == "build.configure"), {})},
            "service": {"desired": self.plan["desired_state"]["service"]["desired"],
                "actual": self.service_actual, "unit_path": str(unit_path), "unit_sha256": unit_hash},
            "v4l": {"desired": self.plan["desired_state"]["virtual_camera"].get("desired"),
                "actual": self.v4l_actual, "owned_configuration": self.v4l_owned_configuration},
            "models": {"packs": model_manifest},
            "ownership": {"application_files": [{"path": str(target / "bin" / name), "type": "file", "creating_operation": "payload.stage", "namespace": "studiocast"} for name in BINARIES] if active else [],
                "links": [{"path": str(self.r["bin"] / name), "type": "symlink", "target": str(self.r["data"] / "current/bin" / name), "creating_operation": "links.reconcile", "namespace": "studiocast"} for name in BINARIES] if active else [],
                "system_configuration": [{**item, "type": "file", "namespace": "studiocast",
                    "creating_operation": "v4l.persistence.write"} for item in self.v4l_owned_configuration],
                "shared_dependencies_observed": []},
            "preservation": {"preserve_settings": self.plan["preservation"]["preserve_settings"],
                "restored": any(item["operation_id"] in {"settings.restore", "settings.restore.compensation"} and
                                item["state"] == "completed" for item in self.journal)},
            "health": {"core": "healthy" if active else "absent", "overall": state,
                "validation": self.plan["expected_validation"], "failure": None if failure is None else {"code": failure.code, "message": failure.message}},
            "journal": self.journal}
        atomic_write(self.manifest_path, manifest)


def validate_operation_contract(plan: dict[str, Any], op: dict[str, Any]) -> None:
    inputs = op.get("inputs")
    if not isinstance(inputs, dict):
        raise BackendError("plan.operation_schema", f"Operation inputs must be an object: {op.get('id')}")
    kind = op["kind"]
    if kind == "trusted_helper":
        operation_type = inputs.get("operation_type")
        expected = PRIVILEGED_INPUT_FIELDS.get(operation_type)
        if expected is None or set(inputs) != set(expected):
            raise BackendError("plan.unsafe_privilege", f"Privileged operation schema mismatch: {op.get('id')}")
        if operation_type == "packages.ensure.v1":
            packages = inputs.get("packages")
            if not isinstance(packages, list) or not packages or any(not isinstance(item, str) for item in packages):
                raise BackendError("plan.unsafe_privilege", "Privileged packages must be a non-empty string array")
        elif operation_type == "v4l.persistence.remove.v1":
            hashes = inputs.get("expected_hashes")
            if (not isinstance(hashes, dict) or set(hashes) != set(V4L_OWNED_PATHS) or
                    any(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value)) is None for value in hashes.values())):
                raise BackendError("plan.unsafe_privilege", "Persistence removal requires both reviewed hashes")
        else:
            if (type(inputs.get("device_number")) is not int or not 0 <= inputs["device_number"] <= 63 or
                    type(inputs.get("exclusive_caps")) is not bool or
                    re.fullmatch(r"[ A-Za-z0-9_.()+-]{1,64}", str(inputs.get("label", ""))) is None):
                raise BackendError("plan.unsafe_privilege", "Invalid reviewed v4l arguments")
    elif kind == "cleanup" and inputs.get("target") not in {
            "payloads", "build_cache", "runtime_state", "models", "settings", "cache", "data", "service"}:
        raise BackendError("plan.operation_schema", f"Unknown cleanup target: {inputs.get('target')}")
    elif kind == "release_acquire":
        expected_fields = {"manifest_path", "signature_path", "manifest_sha256", "manifest_key_id",
                           "version", "artifact", "candidate_path", "cache_dir", "offline"}
        artifact = inputs.get("artifact")
        if set(inputs) != expected_fields or not isinstance(artifact, dict):
            raise BackendError("plan.operation_schema", "Release acquisition contract is incomplete")
        if (re.fullmatch(r"[0-9a-f]{64}", str(inputs.get("manifest_sha256", ""))) is None or
                type(inputs.get("offline")) is not bool or
                re.fullmatch(r"[A-Za-z0-9._-]{1,128}", str(artifact.get("filename", ""))) is None or
                re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", ""))) is None or
                type(artifact.get("size_bytes")) is not int or artifact["size_bytes"] <= 0):
            raise BackendError("plan.operation_schema", "Release acquisition metadata is invalid")
        validate_mutation_path(Path(inputs["cache_dir"]), [roots()["cache"]])
        expected_source = Path(inputs["cache_dir"]) / artifact["filename"]
        if expected_source != Path(plan.get("source", {}).get("path", "")):
            raise BackendError("plan.path_binding", "Release cache target differs from the reviewed source")
    elif kind == "model_transaction":
        expected_fields = {"pack_id", "family", "catalog_version", "catalog_sha256",
                           "source", "offline", "retries", "recommended", "destination_root", "pack"}
        if (set(inputs) != expected_fields or inputs.get("pack_id") not in DEFAULT_MODEL_PACKS or
                inputs.get("family") not in {"open_audio", "open_video"} or
                re.fullmatch(r"sha256:[0-9a-f]{64}", str(inputs.get("catalog_sha256", ""))) is None or
                not isinstance(inputs.get("source"), str) or not inputs["source"] or
                type(inputs.get("offline")) is not bool or type(inputs.get("retries")) is not int or
                type(inputs.get("recommended")) is not bool or
                not 0 <= inputs["retries"] <= 10 or not isinstance(inputs.get("pack"), dict) or
                inputs["pack"].get("id") != inputs["pack_id"] or
                inputs["pack"].get("family") != inputs["family"]):
            raise BackendError("plan.operation_schema", "Curated model transaction contract is invalid")
        validate_mutation_path(Path(inputs["destination_root"]),
                               [roots()["data"] / "models"], allow_root=True)
    elif kind == "build_configure":
        expected_cmake = {"STUDIOCAST_ENABLE_OPEN_CUDA", "STUDIOCAST_ENABLE_OPEN_AUDIO",
                          "STUDIOCAST_ENABLE_OPEN_VULKAN"}
        if set(inputs.get("cmake", {})) != expected_cmake or any(type(value) is not bool for value in inputs["cmake"].values()):
            raise BackendError("plan.operation_schema", "CMake feature flags must be explicit booleans")
        if inputs.get("build_path") != plan.get("paths", {}).get("build_cache"):
            raise BackendError("plan.path_binding", "Configure build path differs from reviewed path")
    elif kind == "build_compile":
        if inputs.get("build_path") != plan.get("paths", {}).get("build_cache") or tuple(inputs.get("targets", ())) != BINARIES:
            raise BackendError("plan.operation_schema", "Build operation targets or path differ from the reviewed contract")
    elif kind == "payload_stage" and inputs.get("build_path") != plan.get("paths", {}).get("build_cache"):
        raise BackendError("plan.path_binding", "Payload stage build path differs from reviewed path")
    elif kind == "payload_validate" and tuple(inputs.get("required_binaries", ())) != BINARIES:
        raise BackendError("plan.operation_schema", "Payload validation binary set is incomplete")
    elif kind == "links_reconcile":
        names = inputs.get("binaries", list(BINARIES))
        if tuple(names) != BINARIES:
            raise BackendError("plan.operation_schema", "Link reconciliation binary set differs from the owned set")


def validate_current_preconditions(plan: dict[str, Any]) -> None:
    r = roots()
    paths = plan.get("paths", {})
    expected_roots = {
        "build_cache": r["cache"], "staging_payload": r["data"],
        "target_payload": r["data"] / "payloads", "current_link": r["data"],
    }
    for name, root in expected_roots.items():
        value = paths.get(name)
        if not isinstance(value, str):
            raise BackendError("plan.path_binding", f"Reviewed path is missing: {name}")
        validate_mutation_path(Path(value), [root])
    classification, manifest, _ = manifest_state(r["data"] / "install-manifest.json")
    reviewed = plan.get("current_state", {})
    if classification != reviewed.get("classification", "absent"):
        raise BackendError("plan.precondition_changed", "Installation classification changed after review")
    actual_active = None
    if manifest:
        actual_active = (manifest.get("versions", {}).get("active") if manifest.get("schema_version") == 2
                         else manifest.get("installed_version"))
    if actual_active != reviewed.get("active_version"):
        raise BackendError("plan.precondition_changed", "Active installation version changed after review")
    if any(op.get("privilege") == "trusted_helper" for op in plan.get("operations", [])):
        test_mode = os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1"
        if not test_mode:
            os_now = os_facts()
            expected = plan.get("privilege_preconditions", {})
            if (os_now["base_id"] != expected.get("base_os") or
                    os_now["base_version_id"] != expected.get("base_release") or
                    os_now["kernel_release"] != expected.get("kernel_release")):
                raise BackendError("plan.precondition_changed", "Privileged host preconditions changed after review")
        helper_now = helper_facts()
        if not (helper_now.get("trusted") and helper_now.get("compatible") and
                helper_now.get("authorization_transport_available")):
            raise BackendError(helper_now.get("reason_codes", ["privileged_helper_missing"])[0],
                               "Trusted helper changed or authorization is unavailable after review")
        expected_helper = plan.get("privilege_preconditions", {}).get("helper", {})
        for key in ("helper_version", "protocol_versions"):
            if expected_helper.get(key) is not None and helper_now.get(key) != expected_helper.get(key):
                raise BackendError("plan.precondition_changed",
                                   "Trusted helper compatibility changed after review")


def validate_model_plan_contract(plan: dict[str, Any]) -> None:
    model_ops = [op for op in plan.get("operations", [])
                 if op.get("kind") == "model_transaction"]
    downloads = plan.get("downloads")
    disk = plan.get("disk")
    if not isinstance(downloads, list) or not isinstance(disk, dict):
        raise BackendError("plan.operation_schema", "Reviewed download/disk contract is incomplete")
    if any(type(item.get("size")) is not int or item["size"] <= 0
           for item in downloads if isinstance(item, dict)):
        raise BackendError("plan.operation_schema", "Reviewed downloads require positive exact sizes")
    if any(not isinstance(item, dict) for item in downloads):
        raise BackendError("plan.operation_schema", "Reviewed downloads must be objects")
    download_bytes = sum(item["size"] for item in downloads)
    required_fields = {"free_bytes", "base_required_bytes", "download_bytes", "required_bytes"}
    if (set(disk) != required_fields or
            any(type(disk.get(field)) is not int or disk[field] < 0 for field in required_fields) or
            disk["download_bytes"] != download_bytes or
            disk["required_bytes"] != disk["base_required_bytes"] + download_bytes):
        raise BackendError("plan.operation_schema", "Reviewed disk totals do not match downloads")
    if not model_ops:
        if any("pack_id" in item for item in downloads):
            raise BackendError("plan.operation_schema", "Model downloads have no reviewed transaction")
        return

    _, catalog, _, catalog_hash = load_model_catalog()
    catalog_packs = {pack["id"]: pack for pack in catalog["packs"]}
    inputs_by_id = {op["inputs"]["pack_id"]: op["inputs"] for op in model_ops}
    desired_ids = plan.get("desired_state", {}).get("model_pack_ids")
    if (not isinstance(desired_ids, list) or len(desired_ids) != len(set(desired_ids)) or
            set(desired_ids) != set(inputs_by_id) or len(inputs_by_id) != len(model_ops)):
        raise BackendError("model.catalog.changed", "Reviewed model selection is not canonical")
    if (catalog["size_metadata"].get("trusted") is not True or
            not catalog["size_metadata"].get("reason_code")):
        raise BackendError("artifact_sizes_untrusted", "Packaged model sizes are not trusted")
    expected_downloads: dict[str, dict[str, Any]] = {}
    for pack_id in desired_ids:
        current = catalog_packs.get(pack_id)
        inputs = inputs_by_id[pack_id]
        if (current is None or inputs.get("catalog_sha256") != catalog_hash or
                inputs.get("catalog_version") != catalog["catalog_version"] or
                canonical_bytes(inputs.get("pack")) != canonical_bytes(current) or
                inputs.get("family") != current["family"]):
            raise BackendError("model.catalog.changed",
                               "Packaged model catalog differs from the reviewed plan")
        for artifact in current["artifacts"]:
            if (type(artifact.get("size_bytes")) is not int or artifact["size_bytes"] <= 0 or
                    artifact.get("size_status") != "known"):
                raise BackendError("artifact_sizes_untrusted",
                                   "Packaged model sizes are incomplete or inconsistent")
            artifact_id = f"{pack_id}:{artifact['name']}"
            expected_downloads[artifact_id] = {
                "pack_id": pack_id, "filename": artifact["name"],
                "size": artifact["size_bytes"], "size_status": "known",
                "sha256": "sha256:" + artifact["sha256"],
                "source_paths": artifact["source_paths"],
            }
    reviewed_model_downloads = {item.get("artifact_id"): item for item in downloads
                                if "pack_id" in item}
    if (len(reviewed_model_downloads) != len(expected_downloads) or
            sum("pack_id" in item for item in downloads) != len(expected_downloads) or
            set(reviewed_model_downloads) != set(expected_downloads)):
        raise BackendError("model.catalog.changed", "Reviewed model downloads are incomplete")
    for artifact_id, expected in expected_downloads.items():
        reviewed = reviewed_model_downloads[artifact_id]
        if any(reviewed.get(key) != value for key, value in expected.items()):
            raise BackendError("model.catalog.changed", "Reviewed model download metadata changed")


def validate_plan(plan: dict[str, Any], supplied_digest: str, token: str, facts: dict[str, Any] | None) -> None:
    if plan.get("schema_version") != 1 or plan.get("plan_version") != PLAN_VERSION or plan.get("policy_version") != POLICY_VERSION:
        raise BackendError("plan.unsupported_schema", "Unsupported plan schema or policy")
    try:
        parsed_plan_id = uuid.UUID(str(plan.get("plan_id")))
    except ValueError as exc:
        raise BackendError("plan.invalid_id", "Reviewed plan ID is invalid") from exc
    if str(parsed_plan_id) != plan.get("plan_id"):
        raise BackendError("plan.invalid_id", "Reviewed plan ID must use canonical UUID syntax")
    if re.fullmatch(r"[0-9a-f]{64}", token or "") is None:
        raise BackendError("plan.token_invalid", "Plan approval token has invalid syntax")
    if not isinstance(plan.get("operations"), list) or not plan["operations"]:
        raise BackendError("plan.operation_schema", "Reviewed plan operations must be a non-empty array")
    if not isinstance(plan.get("blockers"), list):
        raise BackendError("plan.operation_schema", "Reviewed plan blockers must be an array")
    calculated = plan_digest(plan)
    if supplied_digest != calculated or plan.get("plan_digest") != calculated:
        raise BackendError("plan.digest_mismatch", "Reviewed plan digest does not match")
    try:
        expires = parse_time(plan["expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError("plan.invalid_expiry", "Reviewed plan expiry is invalid") from exc
    if expires < dt.datetime.now(dt.timezone.utc):
        raise BackendError("plan.expired", "Reviewed plan has expired")
    pending = load_json(pending_path(plan["plan_id"]))
    if pending.get("used") or pending.get("uid") != os.getuid() or pending.get("digest") != calculated or pending.get("token") != token:
        raise BackendError("plan.token_invalid", "Plan approval token is missing, used, or invalid")
    if facts is not None and facts.get("host_fingerprint") != plan.get("facts_fingerprint"):
        raise BackendError("plan.precondition_changed", "Machine facts changed after review")
    validate_current_preconditions(plan)
    if plan.get("blockers"):
        raise BackendError("plan.blocked", "Plan contains unresolved blockers", {"blockers": plan["blockers"]})
    known: set[str] = set()
    for op in plan.get("operations", []):
        if op.get("kind") not in ALLOWED_OPERATION_KINDS or op.get("privilege") not in {"none", "user", "trusted_helper"}:
            raise BackendError("plan.unknown_operation", f"Unsupported plan operation: {op.get('id')}")
        op_id = op.get("id")
        if not isinstance(op_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", op_id) is None or op_id in known:
            raise BackendError("plan.operation_id", f"Invalid or duplicate operation ID: {op_id}")
        if not isinstance(op.get("depends_on"), list) or any(dep not in known for dep in op["depends_on"]):
            raise BackendError("plan.dependency_order", f"Invalid dependency order for {op.get('id')}")
        if op.get("privilege") == "trusted_helper":
            inputs = op.get("inputs", {})
            if inputs.get("operation_type") not in PRIVILEGED_TYPES or any(key in FORBIDDEN_HELPER_FIELDS for key in inputs):
                raise BackendError("plan.unsafe_privilege", f"Unsafe privileged operation: {op.get('id')}")
        validate_operation_contract(plan, op)
        known.add(op_id)
    validate_model_plan_contract(plan)
    pending["used"] = True
    atomic_write(pending_path(plan["plan_id"]), pending)
