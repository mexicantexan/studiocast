# StudioCast installer backend audit and contract proposal

Date: 2026-07-12
Audited baseline: `5f3bc485a4dbb2d3a463945591f910fbab53c9a6`
Scope: the current shell backend, its delegated install/uninstall helpers, and
the backend-, wizard-, and CMake-cache-facing tests. This is a Wave 1 contract
document; it intentionally changes no production code.

## Policy constraints applied

This proposal preserves the repository's existing authority and compatibility
rules:

- The installer GUI remains a controller. It reviews a backend-produced plan
  and reports backend results; it does not independently decide what happened.
- Existing effect IDs, CLI flags, CMake identifiers, and the Open Video/Open
  CUDA compatibility names are not renamed.
- Maxine assets are never redistributed. A directory's existence is not proof
  that a Maxine component is usable.
- Models remain XDG user data, use safe pack-relative artifact paths, expose
  license/provenance, and are never fetched by the live daemon.
- Runtime/provider/model discovery is analysis/setup work, not frame-loop work.
- Recommended installation is user-local. Host-level virtual-camera work is a
  separately authorized integration operation.

The redesign requirements supersede two descriptions of current MVP behavior
in `docs/DEVELOPER_GUIDE.md`: production binaries must no longer point into the
build cache, and clean install no longer preserves models/logs/cache. It
preserves only settings when the single `preserve_settings` choice is true.

## Evidence-backed findings

### 1. Manifest handling is line-oriented and fail-open

Confirmed.

- `manifest_field` at
  `installer/backend/studiocast-installer-backend:253-258` applies a `sed`
  expression to individual lines. It does not parse JSON, validate a schema,
  distinguish `null` from missing, handle reordered/nested values, reject
  duplicate keys, or report corrupt/truncated input.
- Status reads `installed_version`, `state`, `source_path`, and `build_path`
  through that helper at lines 561-609. At lines 595-598, any present manifest
  whose parsed `state` is not exactly `uninstalled` can count as installed when
  the `studiocast` link is executable. An empty state from corrupt JSON is
  therefore treated like an installed state.
- `write_manifest` at lines 1111-1175 writes schema v1 manually. The temporary
  filename is fixed, there is no file or directory `fsync`, no transaction
  journal, and no representation of desired versus actual state, source
  verification, owned-file hashes, versioned payloads, rollback, or incomplete
  operations.
- `manifest` at lines 1375-1380 merely cats the file. Unknown schemas and
  invalid JSON are not classified.

Existing coverage checks only that status text contains selected optional
component keys (`tests/installer_backend_tests.cpp:322-350`). There are no
tests for valid JSON parsing, v1 migration, truncation, corrupt JSON, duplicate
keys, unknown schema, atomic persistence, stale ownership paths, or failed
transactions.

### 2. The review plan is descriptive output, not the execution input

Confirmed.

- `plan_workflow` at lines 690-875 assembles prose and rendered shell command
  strings. `print_plan_json` at lines 906-918 has no schema/policy version,
  operation objects, dependencies, preconditions, artifact metadata, digest,
  token, or expiry.
- `run` never consumes that JSON. `main` dispatches directly to
  `run_workflow` at lines 1382-1387, which reconstructs behavior from freshly
  parsed CLI flags. A reviewed plan can be stale or tampered with because there
  is no plan validation path at all.
- The plan lists backend `detect-os` and `status` commands at lines 821-822,
  but execution runs neither command. Execution calls the internal
  `ensure_supported_os` only for install-like work.
- Conversely, execution performs archive extraction (`prepare_source`), fresh
  build removal (`configure_build`), daemon configuration, model side effects,
  manifest writing, and clean-install uninstall without representing all of
  them as exact operation objects. For `clean-install`, the command list enters
  the common install-like case and does not list the uninstall helper at all.
- The human prose can claim files or data will be removed without an executable
  operation corresponding to each item. Commands are review strings, not an
  allowlisted execution contract.

The GUI currently calls `plan ... --json` and then later constructs a new
workflow invocation from widget state
(`installer/gui/installer_wizard.cpp:529-539,541-623`). The two argument lists
are similar, but there is no immutable reviewed object binding them. The CLI
has the same flaw because `plan` and `run` are separate reconstructions.

Existing tests assert fragments of rendered commands and prose. None records
operation IDs during execution and compares them with the reviewed operation
list; none alters a plan or a relevant precondition and expects rejection.

### 3. v4l2 selection is lost when the dependency bundle is skipped

Confirmed independently with a hermetic plan invocation using temporary
HOME/XDG paths:

```text
plan install --skip-deps --v4l2loopback --load-loopback --persist-loopback
```

The plan claimed all three v4l changes and a privileged v4l operation but its
commands contained no `scripts/setup.sh` invocation.

The defect is shared by planning and execution:

- The planner's setup command gate at lines 825-838 considers only
  `WITH_DEPS`, Vulkan runtime, and shader tools. `WITH_V4L2` alone cannot enter
  it.
- `run_setup_deps` has the same early return at lines 993-996. Its v4l
  arguments at lines 999-1003 are unreachable in the v4l-only case.
- `scripts/setup/ubuntu.sh:348-349` makes `--v4l2loopback` imply its own needed
  dependency work, so the backend should issue that bounded operation even
  when the general dependency bundle is deselected.

The existing Vulkan-only plan test catches the analogous Vulkan case
(`tests/installer_backend_tests.cpp:199-223`) but there is no v4l-only test.

### 4. CMake choices are not explicit OFF values

Confirmed independently with a hermetic plan invocation using
`--no-open-backends --no-open-vulkan`: the CMake command contained none of the
three feature variables.

- Planning adds only `...OPEN_CUDA=ON`, `...OPEN_AUDIO=ON`, and
  `...OPEN_VULKAN=ON` at lines 839-845.
- Execution repeats the same ON-only construction at lines 1028-1034.
- A reused cache can therefore retain a previous ON value after the reviewed
  selection changes to OFF (or retain an OFF value for an option not explicitly
  asserted by a future selection).

The current negative test explicitly expects disabled features merely to omit
the ON flags (`tests/installer_backend_tests.cpp:252-274`). The separate CMake
cache suite validates CMake project behavior, but not an ON-to-OFF or OFF-to-ON
installer reconfiguration of the same cache
(`tests/cmake_open_backend_cache_tests.cpp:92-224`).

### 5. Manifest defaults resolve after review, and only for repair paths

Confirmed.

- Backend defaults are process globals at lines 36-65. Every workflow starts
  with service on, v4l on, general dependencies on, models off, open backends
  on, Vulkan off, and the generic cache build path.
- `plan_workflow` computes source, target, and build summary from those globals
  at lines 694-716.
- Only after execution begins does repair read v1 `source_path` and
  `build_path`, at lines 1225-1239. The reviewed repair plan can therefore show
  and hash different paths from the ones execution uses.
- Update does not reuse manifest source/build values at all. Neither update nor
  repair restores desired service, v4l, models, providers/backends, build type,
  preservation, or model selections because v1 does not record them.
- A missing or corrupt manifest silently falls back to current defaults rather
  than producing a blocker/reconstruction state.

There is no installed-configuration reuse test. Existing repair tests build
their expectation from hard-coded defaults.

### 6. Production binaries are symlinked directly into disposable build state

Confirmed.

- `install_binaries_and_service` delegates the selected build directory to
  `scripts/install.sh user-service` at backend lines 1043-1050.
- `scripts/install/user_service.sh:135-161` creates each `~/.local/bin` link
  directly to `${BUILD_DIR}/${binary}`.
- The default build directory is `${XDG_CACHE_HOME}/studiocast/build` at
  backend lines 31-38. `--fresh-build` removes an allowed cache directory at
  lines 1013-1023. Clean install forces fresh build at line 1246.
- Removing the cache or an extracted source/build tree therefore breaks the
  active commands and service. There is no staged runtime payload, atomic
  `current` link, previous version, activation validation, or rollback.

The only user-service test checks enable/restart command text
(`tests/installer_backend_tests.cpp:471-513`). No test removes the build cache
and then executes the installed command.

### 7. Failure and partial state cannot be represented accurately

Confirmed.

- The backend uses `set -e` and sequential shell calls. It returns a numeric
  process failure but has no structured result object or journal.
- Install order is build, daemon config mutation, binary/service linking,
  model download, then manifest write (lines 1202-1217). If an optional model
  install fails, core binaries and service may already be active while a first
  install has no manifest. Status then reports not installed because it requires
  both manifest and executable link (lines 595-598).
- During update, binary symlinks are replaced before optional model work and
  before the manifest is updated. A later failure can leave the new binaries
  active with the old manifest; there is no way to preserve the prior active
  payload.
- Model installs are two sequential aggregate commands (lines 1052-1070). A
  failure cannot be recorded per pack/artifact or retried from a committed core
  result.
- Service enablement treats missing `systemctl` as successful helper completion
  (`scripts/install/user_service.sh:217-227`), while the manifest only samples
  strings afterward. Desired and actual state are not separated.
- Uninstall's service stop/disable/reload calls are best effort and suppress
  failures (`scripts/uninstall/uninstall.sh:117-132`); the backend can write an
  `uninstalled` tombstone without preserving these partial failures.
- There is no cancellation contract, child-process group tracking, rollback
  handler, or distinction among `failed`, `cancelled`, `degraded`, and
  `committed`.

No existing backend test injects configure/build/install/model/service failure,
cancellation, or rollback and then verifies status and manifest state.

### 8. Deletion and ownership evidence is insufficient

Confirmed.

- Full user-data removal uses only broad XDG prefix strings and `rm -rf`
  (`remove_user_data_dirs`, lines 1177-1187). This is not manifest-path-driven,
  which avoids one class of stale-manifest deletion, but it is too broad for
  settings-only preservation and has no symlink-component validation.
- Simple uninstall removes known names but does not prove that a binary link,
  service file, or desktop file is StudioCast-owned before deleting it
  (`scripts/uninstall/uninstall.sh:117-168`).
- The v1 manifest records paths and existence but no file type, target, hash,
  ownership class, or creation operation. Shared dependencies are summarized as
  a string, so they cannot be distinguished from owned artifacts.
- The legacy greedy helper can purge shared packages. The redesigned backend
  must never route ordinary uninstall or clean install through that behavior.

Existing package-safety coverage protects Maxine redistribution only. There is
no stale path, symlink traversal, ownership/hash mismatch, or shared dependency
non-removal test.

## Frozen contract proposal

All externally stored documents use UTF-8 JSON, reject duplicate object keys,
use integer schema versions, and preserve unknown fields when reading and
rewriting a supported schema. Paths are absolute lexical-normal paths in
plans/manifests, but every mutation also resolves and validates parent
components at execution time. Reason codes and operation IDs are stable API;
human messages are presentation data and may change.

### Analysis facts v1

The analyzer command accepts `analyze --json` for the real host and
`analyze --input-fixture FILE --json` for hermetic fixtures. Fixture mode must
perform no host probes. The recommendation command consumes the resulting
document and never probes the host.

```json
{
  "schema_version": 1,
  "facts_version": "installer-facts/v1",
  "captured_at": "2026-07-12T00:00:00Z",
  "host_fingerprint": "sha256:<canonical-stable-facts>",
  "os": {
    "id": "ubuntu",
    "version_id": "24.04",
    "base_id": "ubuntu",
    "base_version_id": "24.04",
    "architecture": "x86_64",
    "kernel_release": "6.x",
    "supported": true,
    "reason_codes": ["os.supported.ubuntu_noble"]
  },
  "installation": {
    "classification": "absent",
    "reason_codes": ["install.manifest.absent"],
    "manifest": {"exists": false, "health": "absent", "schema_version": null},
    "active_version": null,
    "previous_version": null,
    "target_version": "0.2.9",
    "version_relation": "not_installed",
    "desired_configuration": null,
    "payloads": []
  },
  "paths": {
    "data_root": {"path": "/home/u/.local/share/studiocast", "writable": true},
    "cache_root": {"path": "/home/u/.cache/studiocast", "writable": true},
    "config_root": {"path": "/home/u/.config/studiocast", "writable": true},
    "free_bytes": 12000000000,
    "required_bytes": {"download": 1, "build": 1, "staging": 1}
  },
  "toolchain": {
    "tools": {},
    "build_dependencies": [],
    "missing_build_dependencies": [],
    "reason_codes": []
  },
  "runtime": {
    "dependencies": {},
    "onnxruntime": {
      "present": false,
      "version": null,
      "compatible": false,
      "providers": {"cpu": false, "cuda": false},
      "probe": "not_run",
      "reason_codes": ["runtime.ort.absent"]
    }
  },
  "systemd_user": {
    "systemctl_present": true,
    "manager_usable": true,
    "reason_codes": ["service.user_manager.usable"]
  },
  "v4l2": {
    "package_state": "missing",
    "module_available": false,
    "module_loaded": false,
    "device_present": false,
    "device_number": 10,
    "device_number_conflict": false,
    "persistence_state": "absent",
    "kernel_headers_usable": true,
    "dkms_viable": true,
    "secure_boot": "unknown",
    "reason_codes": ["v4l.module.absent"]
  },
  "gpus": {
    "devices": [],
    "nvidia": {
      "driver_usable": false,
      "cuda_usable": false,
      "reason_codes": ["gpu.nvidia.absent"]
    },
    "vulkan": {
      "loader_present": false,
      "physical_devices": [],
      "compute_device_usable": false,
      "reason_codes": ["gpu.vulkan.loader_absent"]
    }
  },
  "maxine": {
    "components": {
      "video_fx": {"present": false, "usable": false},
      "ar": {"present": false, "usable": false},
      "audio_fx": {"present": false, "usable": false}
    },
    "effects": {},
    "reason_codes": ["maxine.sdk.absent"]
  },
  "effects": {
    "capabilities": {
      "virtual_background": {
        "maxine": "unavailable",
        "cuda": "unavailable",
        "vulkan": "unavailable",
        "cpu": "unavailable"
      }
    }
  },
  "models": {
    "packs": {},
    "default_pack_ids": [
      "fastenhancer_s_vd_v1",
      "fastenhancer_m_vd_v1",
      "modnet-webnn-256-fp32",
      "yunet_opencv_zoo_2023mar_fp32",
      "dlib_68_ibug_300w",
      "gaze_correction_cam_flx_v0_1_1",
      "fastdvdnet_sigma15"
    ]
  },
  "cache": {
    "release_artifacts": {},
    "model_artifacts": {}
  },
  "connectivity": {
    "release_source": {"state": "offline", "reason_codes": ["network.release.offline"]},
    "model_source": {"state": "offline", "reason_codes": ["network.models.offline"]}
  },
  "reason_codes": []
}
```

Required classifications are `absent`, `healthy`, `partial`, `unhealthy`,
`stale`, `tombstone`, and `unknown_schema`. Probe states distinguish `usable`,
`unusable`, `absent`, `unknown`, and `not_run`; presence alone must never become
usability. Vulkan physical devices include vendor/device IDs, type, compute
queue support, API/driver version, software-device signal, and stable identity.
Maxine usability is recorded per component and per effect after loading/smoke
validation, not from directory checks.

Stable reason-code families include:

- `os.supported.*`, `os.unsupported.*`, `arch.unsupported.*`
- `install.manifest.{absent,healthy,corrupt,truncated,unknown_schema}`
- `install.payload.{missing,corrupt,healthy}`, `install.version.*`
- `disk.{data,cache,staging}.insufficient`, `path.*.not_writable`
- `toolchain.*.missing`, `runtime.*.{absent,incompatible,unusable}`
- `service.user_manager.{usable,unavailable,unreachable}`
- `v4l.{package,module,device,persistence,headers,dkms,secure_boot,number}.*`
- `gpu.{nvidia,cuda,vulkan}.*`, `maxine.*`, `effect.<id>.<engine>.*`
- `model.<pack_id>.{absent,verified,corrupt,license_missing}`
- `cache.<artifact>.{verified,absent,corrupt}` and `network.*.*`

### Recommendation v1

The pure function is:

```text
recommend(facts, intent, prior_desired_configuration, policy_version)
  -> recommendation
```

It rejects a facts schema or policy version it does not support. It does not
read time, environment, filesystem, devices, or network. Repeated canonical
inputs produce byte-identical canonical output.

```json
{
  "schema_version": 1,
  "recommendation_version": "installer-recommendation/v1",
  "policy_version": "studiocast-installer-policy/1",
  "facts_fingerprint": "sha256:...",
  "intent": "install",
  "route": "recommended",
  "primary_action": "install",
  "selections": {
    "scope": "user",
    "source": {"kind": "official_release", "verified_required": true},
    "build_type": "Release",
    "service": {"desired": "enabled_started"},
    "virtual_camera": {"desired": true, "required_for_success": true},
    "model_pack_ids": [],
    "effects": {
      "virtual_background": {"engine_precedence": ["maxine", "cuda", "vulkan", "cpu"], "selected": "unavailable"}
    },
    "shader_tools": false,
    "mesa_packages": false,
    "preserve_settings": true
  },
  "reasons": [{"code": "source.official.required", "applies_to": "source"}],
  "blockers": [],
  "warnings": [],
  "alternatives": []
}
```

For Recommended, `model_pack_ids` is exactly the seven IDs in the facts example
even when Maxine is usable. Per effect, select the first genuinely usable path
in Maxine, CUDA, Vulkan, CPU order. Continue to CPU only when that effect's
capability matrix says CPU is implemented. Loader-only Vulkan is never usable.
Mesa is never recommended for NVIDIA, shader tools are never recommended, and
an unsupported OS yields a blocker before any automatic package/module plan.
Custom may set `virtual_camera.desired=false`, but then emits
`install.degraded.no_virtual_camera` and cannot claim full success. Prior
desired configuration seeds update, repair, and reinstall; intent-specific
policy changes only values the user explicitly changed or that became unsafe.

### Plan v1

The plan is the sole execution input. GUI and CLI both persist the backend's
reviewed plan and invoke `execute-plan --plan FILE --digest sha256:... --token
TOKEN`; they do not reconstruct flags.

```json
{
  "schema_version": 1,
  "plan_version": "installer-plan/v1",
  "policy_version": "studiocast-installer-policy/1",
  "plan_id": "uuid",
  "created_at": "2026-07-12T00:00:00Z",
  "expires_at": "2026-07-12T00:30:00Z",
  "intent": "install",
  "route": "recommended",
  "facts_fingerprint": "sha256:...",
  "current_state": {},
  "desired_state": {},
  "paths": {
    "build_cache": "/home/u/.cache/studiocast/builds/0.2.9",
    "staging_payload": "/home/u/.local/share/studiocast/payloads/.staging-uuid",
    "target_payload": "/home/u/.local/share/studiocast/payloads/0.2.9",
    "current_link": "/home/u/.local/share/studiocast/current"
  },
  "downloads": [],
  "operations": [],
  "preservation": {"preserve_settings": true, "snapshot_path": null},
  "blockers": [],
  "warnings": [],
  "expected_validation": [],
  "plan_digest": "sha256:...",
  "approval_token": "opaque-single-use-token"
}
```

Every download entry contains `artifact_id`, URL, exact size, SHA-256,
signature/key ID where applicable, provenance, license identifier/name/URL,
cache path, destination, and `required_for_core`. Mutable or unhashed core
artifacts are blockers.

Every operation contains:

```json
{
  "id": "payload.activate",
  "kind": "activate_payload",
  "depends_on": ["payload.validate"],
  "inputs": {},
  "source_paths": [],
  "target_paths": [],
  "privilege": "user",
  "authorization_operation": null,
  "reversibility": "reversible",
  "failure_policy": "abort_rollback",
  "preconditions": [],
  "validation": [],
  "review": {"category": "application_files", "message_code": "plan.payload.activate"}
}
```

Allowed privilege values are `none`, `user`, and `trusted_helper`. There is no
`shell`, `sudo`, or arbitrary-script privilege. Helper operations reference a
fixed privilege-protocol operation and validated typed arguments.

Canonical operation IDs, omitted when not applicable, are:

1. `preflight.validate`
2. `settings.snapshot`
3. `source.manifest.verify`
4. `source.archive.obtain`
5. `source.archive.verify`
6. `source.archive.extract`
7. `build.cache.prepare`
8. `dependencies.packages.ensure`
9. `build.configure`
10. `build.compile`
11. `payload.stage`
12. `payload.validate`
13. `v4l.package.ensure`
14. `v4l.module.load`
15. `v4l.persistence.write`
16. `payload.activate`
17. `links.reconcile`
18. `service.unit.reconcile`
19. `service.state.reconcile`
20. `model.pack.<pack-id>.obtain`
21. `model.pack.<pack-id>.verify`
22. `model.pack.<pack-id>.activate`
23. `settings.restore`
24. `payload.rollback` (compensation only)
25. `cleanup.build_cache`
26. `cleanup.runtime_state`
27. `cleanup.models`
28. `cleanup.owned_integration`
29. `manifest.commit`

Uninstall uses `service.state.reconcile`, `service.unit.reconcile`,
`links.reconcile`, `cleanup.runtime_state`, `cleanup.owned_integration`, and
per-payload/model/config removal operations selected by explicit preservation
policy. Repair includes only operations whose facts differ from desired state.
Clean install explicitly includes settings snapshot when selected, all required
cleanup operations, rebuild, all seven model-pack transactions, and settings
restore.

The digest is SHA-256 over RFC 8785-style canonical JSON with `plan_digest` and
`approval_token` omitted. The token is a random, user-private, single-use nonce
stored in a mode-0600 pending-plan record and bound to plan ID, digest, uid, and
expiry. Execution rejects a missing/reused token, digest mismatch, unsupported
schema/policy, changed uid, expired plan, changed facts fingerprint, changed
file/hash/device preconditions, blockers, unknown operation/argument, or a
dependency-order violation. It records operation IDs as they begin/end; the
executed set must be a prefix/topological realization of the reviewed set plus
only explicitly declared compensation operations.

CMake operation inputs always contain explicit booleans, including OFF:

```json
{
  "STUDIOCAST_ENABLE_OPEN_CUDA": false,
  "STUDIOCAST_ENABLE_OPEN_AUDIO": false,
  "STUDIOCAST_ENABLE_OPEN_VULKAN": false
}
```

### Manifest v2 and journal

```json
{
  "schema_version": 2,
  "policy_version": "studiocast-installer-policy/1",
  "transaction": {
    "id": "uuid",
    "state": "committed",
    "intent": "install",
    "started_at": "...",
    "updated_at": "...",
    "completed_at": "...",
    "plan_id": "uuid",
    "plan_digest": "sha256:..."
  },
  "install_scope": "user",
  "versions": {
    "active": "0.2.9",
    "previous": "0.2.8",
    "target": "0.2.9",
    "relation_at_start": "upgrade"
  },
  "payloads": {
    "root": "/home/u/.local/share/studiocast/payloads",
    "current_link": "/home/u/.local/share/studiocast/current",
    "active": {"version": "0.2.9", "path": "/home/u/.local/share/studiocast/payloads/0.2.9", "tree_hash": "sha256:...", "validation": []},
    "previous": {"version": "0.2.8", "path": "/home/u/.local/share/studiocast/payloads/0.2.8", "tree_hash": "sha256:..."},
    "target": null
  },
  "source": {
    "channel": "stable",
    "official": true,
    "release_manifest_url": "https://...",
    "release_manifest_hash": "sha256:...",
    "signing_key_id": "...",
    "archive_url": "https://...",
    "archive_sha256": "...",
    "archive_signature": "..."
  },
  "desired_configuration": {
    "build_type": "Release",
    "features": {},
    "service": {"desired": "enabled_started"},
    "v4l": {"desired": true, "device_number": 10, "label": "StudioCast Camera", "exclusive_caps": true, "persist": true},
    "model_pack_ids": [],
    "preserve_settings": true
  },
  "effective_build_configuration": {"cmake": {}},
  "service": {"desired": "enabled_started", "actual": "active", "unit_path": "...", "unit_sha256": "..."},
  "v4l": {"desired": true, "actual": {}, "owned_configuration": []},
  "models": {"packs": {}},
  "ownership": {
    "application_files": [],
    "links": [],
    "system_configuration": [],
    "shared_dependencies_observed": []
  },
  "preservation": {"preserve_settings": true, "restored": true},
  "health": {"core": "healthy", "overall": "healthy", "validation": []},
  "journal": []
}
```

Owned entries include path, expected type, uid, mode where relevant, SHA-256 or
link target, creating operation, and namespace. Shared dependencies record
package/library identity and observed version only; they are never deletion
authority. Model pack entries contain pack ID, schema/task, license/provenance,
and every one of the eight default artifact files with size and verified hash.

Journal entries are append-semantics records persisted atomically:

```json
{
  "sequence": 12,
  "operation_id": "model.pack.fastdvdnet_sigma15.activate",
  "state": "failed",
  "started_at": "...",
  "finished_at": "...",
  "exit": {"kind": "process_exit", "code": 1, "signal": null},
  "error": {"code": "model.download.checksum_mismatch", "message": "...", "details": {}},
  "compensation_operation_ids": []
}
```

Transaction states are `prepared`, `executing`, `core_committed`, `committed`,
`degraded`, `rolling_back`, `rolled_back`, `failed`, and `cancelled`.
`core_committed` means a validated payload is active even if optional model or
service work remains. Status derives installation presence from payload and
transaction health rather than hiding core installation after optional failure.

Atomic persistence uses a unique same-directory temporary file, mode 0600,
complete serialization and validation, file `fsync`, atomic rename, then parent
directory `fsync`. The previous valid manifest is retained until replacement is
durable. Recovery treats a leftover temporary file only as diagnostic evidence;
it never supersedes the last valid manifest without validation.

### v1 migration and safe deletion

1. Parse v1 as JSON with strict type checks. Never use line extraction.
2. Preserve the original v1 bytes as a read-only migration diagnostic; do not
   rewrite or delete them until v2 is durably committed.
3. Treat v1 `source_path`, `build_path`, binary paths, and service path as
   untrusted hints. Canonicalize them and use them only for analysis/recovery,
   never as deletion authority.
4. Reconstruct owned links/files only by matching fixed StudioCast namespaces,
   expected file type, expected link target shape or trusted payload root, and
   content hash where available. A mismatch is `ownership.mismatch` and is left
   untouched with a warning.
5. Infer prior desired service/v4l/backend choices from validated actual state
   only when evidence is unambiguous. Otherwise mark each field `unknown` and
   require review; do not replace it with workflow defaults.
6. Map v1 `state=uninstalled` to a v2 tombstone. A corrupt/truncated v1 is
   `unhealthy`, and an unknown schema is `unknown_schema`; both block automatic
   destructive work.
7. Commit v2 only as part of a reviewed reconciliation/migration transaction.

Deletion validates that the target is beneath a compiled/derived allowed root,
that no parent component escapes through a symlink, and that type/hash/link
target matches the owned record. Root, home, XDG roots themselves, relative
paths, traversal, mount surprises, and ownership mismatches are rejected.

## Transaction and failure contract

1. Analyze once and create a deterministic recommendation.
2. Resolve all manifest-derived desired settings before plan generation.
3. Create and review the exact plan. Revalidate digest/token/preconditions
   immediately before any mutation.
4. Persist `prepared`, then journal `preflight.validate` and each operation
   transition before/after execution.
5. Build in a version-specific cache. Stage files under the durable data root,
   validate the staged core, then rename it to the final versioned payload.
6. Atomically switch `current` only after validation. Keep the prior payload and
   pointer as rollback data. User-facing links target the durable `current`
   payload, never the build cache.
7. If core work fails before activation, leave the prior active pointer and
   manifest active version unchanged. If activation or post-activation core
   validation fails, atomically restore the prior pointer and journal rollback.
8. After core activation, mark `core_committed`. Model/download failures are
   retryable `degraded` results unless the plan declared that artifact required
   for a selected feature. They do not erase core presence.
9. Service actual-state failure is structured. Recommended success requires a
   usable requested service and required virtual camera; Custom can complete
   degraded only for explicitly accepted optional omissions.
10. On cancellation, send termination to the tracked process group, wait with a
    bounded timeout, kill remaining children if required, perform declared
    compensation, and persist `cancelled`. Never write a committed manifest
    from a cancelled operation.

Execution returns structured JSON even on nonzero exit:

```json
{
  "transaction_id": "uuid",
  "state": "degraded",
  "core_committed": true,
  "active_version": "0.2.9",
  "failed_operation_id": "model.pack.fastdvdnet_sigma15.obtain",
  "error": {"code": "network.models.offline", "message": "...", "details": {}},
  "retryable_operation_ids": ["model.pack.fastdvdnet_sigma15.obtain"],
  "exit": {"kind": "backend_result", "code": 3, "signal": null}
}
```

The GUI maps only `committed` to full success, `degraded` to a limitation page
with Retry/Continue, and `failed`/`cancelled`/`rolled_back` to explicit outcomes.

## Required implementation tests derived from the gaps

The correctness foundation should first add characterization/regression tests:

- Parse valid v1 and v2 JSON; migrate v1; reject corrupt, truncated, duplicate
  key, and unknown-schema documents without destructive fallback.
- Verify atomic write recovery and failed/cancelled journal preservation.
- Assert `--skip-deps` plus selected v4l produces and executes the exact v4l
  operation.
- Record executed operation IDs and assert equality with the reviewed plan;
  reject changed digest, token, expiry, facts, files, and operation arguments.
- Reconfigure one CMake cache ON-to-OFF and OFF-to-ON and assert every relevant
  variable is explicit in plan, invocation, cache, and manifest.
- Seed an installed desired configuration and prove update, repair, and
  reinstall reuse it before review, including service off, v4l choice, Vulkan,
  models, build type, and custom packs.
- Remove the build cache after activation and execute every installed command;
  confirm the service points into the durable active payload.
- Inject build failure before activation and prove the prior payload stays
  active. Inject model failure after activation and prove status is degraded
  with core present and the failed pack retryable.
- Cancel a helper with a child process and prove no child remains and no commit
  is recorded.
- Reject unsafe/traversal/symlink paths and leave hash/type-mismatched owned
  paths untouched. Confirm shared dependencies are never removed.
- For clean install, prove checked preservation snapshots/restores only user
  config while payload, service, cache, state/logs, and models are reset; prove
  unchecked preservation performs a literal full wipe.

All tests must use isolated HOME/XDG roots, fixture facts/manifests, fake PATH
tools, local artifact sources, and mock privilege transport. No test may invoke
real sudo/pkexec, apt, modprobe, systemctl mutation, network, or model download.

## Implementation boundaries and caveats

- A shell script cannot robustly supply strict JSON/schema validation,
  canonical digesting, durable journaling, semantic version handling, secure
  process supervision, and atomic transactions without becoming another
  fragile parser. The backend core should move to a compiled implementation
  using the repository's JSON facilities; `scripts/installer.sh` can remain a
  compatibility launcher.
- Host-wide operations must be typed requests to the separately designed,
  trusted privilege helper. This document deliberately does not bless invoking
  any setup script from a selected source as root.
- The exact Vulkan per-effect capability values are intentionally delegated to
  the focused Vulkan audit. The schema consumes that matrix and fails closed;
  it does not assume loader/device presence equals production capability.
- Release signatures, installer self-update transport, and the concrete public
  key/signature format are owned by the release/update contract. Plan and
  manifest fields above are designed to carry those verified results.
