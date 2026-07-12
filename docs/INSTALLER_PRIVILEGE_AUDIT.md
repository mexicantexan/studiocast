# Installer Privilege and AppImage Trust-Boundary Audit

Date: 2026-07-12  
Audited baseline: `5f3bc485a4dbb2d3a463945591f910fbab53c9a6`

## Outcome

Both reported defects are present.

1. The packaged backend cannot run its uninstall implementation. It derives a
   repository root from the backend's installed path and looks for
   `scripts/uninstall.sh` outside the staged AppDir contents.
2. The graphical install path does not have a safe or complete privilege
   boundary. The installer delegates to setup scripts from the selected source,
   those scripts call ordinary `sudo`, and the installer GUI neither provides a
   terminal nor implements authorization. Other StudioCast GUI paths have a
   `sudo -S` password shim, but that shim sends a password to an entire
   user-controlled process tree and is not a suitable installer security
   boundary.

The safe design requires two distinct changes:

- Make ordinary user-local uninstall self-contained in the installed backend or
  stage a dedicated user-level uninstaller beside it.
- Move every host-level mutation behind a separately installed, root-owned,
  allowlisted helper with polkit authorization. The AppImage can contain the
  client and protocol, but it cannot safely bootstrap or serve as the privileged
  authority itself.

Until that root-owned helper package is installed, Recommended installation
must stop before mutation when virtual-camera reconciliation is needed. It must
not fall back to `sudo`, `sudo -S`, `SUDO_ASKPASS`, `pkexec` of an AppImage
payload, or execution of a selected source script as root.

## Scope and constraints

This audit covered:

- `installer/backend/studiocast-installer-backend`
- `installer/gui/*`
- the related setup, install, v4l2loopback, and uninstall scripts
- CMake installer-component rules
- AppImage/AppDir build and verification scripts
- the main GUI's setup-repair and virtual-camera recovery paths
- existing installer and packaging tests
- the installer, GUI, model, pipeline, and latency policies in `docs/`

No package-manager, authorization, module, service, network, or real
installation command was run. The only execution reproduction used a temporary
AppDir-shaped directory and temporary `HOME`; it failed before any removal
implementation could start.

## Evidence

### Packaged uninstall is not self-contained

The backend computes:

```text
SCRIPT_DIR = .../usr/share/studiocast/installer
REPO_ROOT  = SCRIPT_DIR/../.. = .../usr/share
```

See `installer/backend/studiocast-installer-backend:9-11`. Both the review plan
and execution then use `${REPO_ROOT}/scripts/uninstall.sh`; see lines 867-869
and 1189-1194.

The Installer CMake component installs only the GUI executable and backend
script (`CMakeLists.txt:905-918`). `packaging/appimage/build_appimage.sh` adds
desktop metadata, icons, `AppRun`, and the source archive, but not an extracted
uninstall helper. `packaging/appimage/verify_bundle.sh` verifies the GUI,
backend, metadata, and source archive, but never requires or executes a
packaged uninstall implementation.

An AppDir-layout reproduction copied the backend to its real installed
location and invoked it with a temporary `HOME`. The plan contained:

```text
<temp>/AppDir/usr/share/scripts/uninstall.sh --yes
```

Execution exited 127 with:

```text
<temp>/AppDir/usr/share/scripts/uninstall.sh: No such file or directory
```

The bundled source archive does contain the script, but it is not extracted at
the path used by uninstall. Uninstall should not depend on a build source or
cache in any case.

### The installer GUI has no complete authentication protocol

The desktop entry explicitly has `Terminal=false`
(`packaging/appimage/studiocast-installer.desktop.in:7`). The installer launches
the backend as a normal `QProcess` and only streams stdout/stderr
(`installer/gui/installer_wizard.cpp:1401-1439`). It does not set the
`STUDIOCAST_GUI_SUDO_STDIN` shim variables, detect a password prompt, write a
password, launch a graphical authorization agent, or classify authorization
errors.

The backend runs the selected source's `scripts/setup.sh`
(`installer/backend/studiocast-installer-backend:987-1009`). That setup script
uses ordinary `sudo` for apt, module, configuration, `/opt`, pkg-config, and
`ldconfig` changes. Therefore the desktop flow works only accidentally when
authorization is already cached or configured passwordlessly. Otherwise it has
no interaction path and fails as raw process output.

The conditional `sudo -S` wrapper in `scripts/setup/ubuntu.sh:88-92` and
`scripts/setup/v4l2loopback.sh:47-51` does not fix this path: the installer does
not enable it and does not feed its stdin.

### The existing `sudo -S` GUI shim is not a safe substitute

The main StudioCast GUI contains two related password-pipe implementations:

- Engine setup repair sets the shim environment, looks for human sudo prompt
  text on stderr, opens a password dialog, and writes the password to the
  backend process stdin (`src/gui/pages/engines_models_page.cpp:1341-1358` and
  1417-1498).
- Virtual-camera recovery runs an in-memory `bash -c` script containing
  `sudo -S`, recognizes prompt substrings, and writes the password to the shell
  stdin (`src/gui/pages/advanced_page.cpp:179-220`, 1078-1091, and 1224-1311).

This protocol is incomplete and unsafe for installer use:

- Prompt recognition depends on localized/human stderr text rather than a
  structured authorization result.
- Password data is written to the stdin of the whole backend or shell. Any
  program in that process tree can read or retain it.
- The engine repair backend can be resolved from an environment override,
  compile-time source, manifest source, or current directory
  (`src/gui/pages/engines_models_page.cpp:1149-1174`).
- The installer backend separately permits a source directory or archive and
  delegates setup to executable scripts within it. A selected source can
  therefore control code adjacent to the password and root commands.
- Cancellation and timeout semantics are process-oriented, not authorization-
  or transaction-oriented. There is no typed distinction between unavailable,
  denied, cancelled, timed out, and partially executed authorization.

The static recovery shell happens not to interpolate user input today, but it
still establishes the wrong boundary: the GUI is handling an account password
and a general shell owns the privileged sequence.

### Selected-source scripts currently control privileged work

Both the plan and executor refer to `${plan_source_dir}/scripts/setup.sh` or
`${EFFECTIVE_SOURCE_DIR}/scripts/setup.sh`. Source selection includes local
directories and arbitrary archives. Checking only for executable files does not
establish trust or provenance.

Consequences include:

- Selected code controls all `sudo` invocations and can add arbitrary ones.
- Selected code controls package-manager arguments and root-written file
  content.
- The ONNX Runtime path downloads an archive without a declared hash and then
  invokes `sudo tar` to extract it under `/opt/studiocast`; it also writes
  system linker and pkg-config files (`scripts/setup/ubuntu.sh:119-190`). A
  root extractor must never consume an archive merely because a user-level
  script selected or downloaded it.
- v4l device number, label, and `exclusive_caps` are not strictly validated
  before becoming module arguments and root-owned configuration. In particular,
  newline/control characters in a label can change generated config syntax.

Builds may still use selected source in Advanced mode, but all compilation and
source-controlled scripts must remain unprivileged. The privileged helper must
not accept a source path, build path, archive, URL, script, command, shell text,
environment, or working directory.

### v4l persistence is inconsistent and not fully namespaced

`scripts/setup/ubuntu.sh` writes:

- `/etc/modules-load.d/v4l2loopback.conf`
- `/etc/modprobe.d/studiocast-v4l2loopback.conf`

The dedicated v4l helper instead uses
`/etc/modules-load.d/studiocast-v4l2loopback.conf`. The non-namespaced first path
can overlap another administrator or application's ownership. Current greedy
uninstall tries to infer ownership from one exact line, but this is not enough
for the new manifest ownership contract.

The new implementation must use exactly these StudioCast-namespaced paths:

```text
/etc/modules-load.d/studiocast-v4l2loopback.conf
/etc/modprobe.d/studiocast-v4l2loopback.conf
```

Legacy `/etc/modules-load.d/v4l2loopback.conf` may be reported and migrated only
when its known content and recorded hash establish StudioCast ownership. It must
otherwise be left untouched with manual instructions.

## Required trust boundary

### Components

```text
unprivileged installer/backend
  - analyzes facts
  - creates and displays the exact plan
  - builds selected source without privilege
  - performs user-local payload/service/model work
  - sends only privileged plan operations to the helper client

polkit authorization agent
  - authenticates/authorizes the desktop user
  - never discloses the password to StudioCast

root-owned studiocast-system-helper
  - parses a small versioned JSON protocol
  - validates every operation and precondition
  - performs only compiled-in operations with absolute executable paths
  - emits structured results and file hashes
```

Recommended packaging target:

```text
/usr/libexec/studiocast/studiocast-system-helper
/usr/share/polkit-1/actions/com.studiocast.Installer1.policy
```

The helper must be a regular root-owned file, mode 0755 or stricter, not
group/world writable, and supplied by a separately installed, signed OS package
(for the initial supported Ubuntu policy, a `studiocast-system-helper` package).
The polkit action must name the exact root-owned executable. A system D-Bus
service with method-level polkit checks is also acceptable if it preserves the
same operation contract and ownership requirements.

The helper must not execute from an AppImage mount, build directory, XDG user
directory, selected source, extracted archive, manifest path, environment
override, or current working directory. Even a hash-checked AppImage copy is
user-writable and has a replacement/TOCTOU boundary when elevated.

### Bootstrap prerequisite

An AppImage alone cannot safely create its own trusted root helper. Doing so
would require elevating a user-controlled executable or using a broad bootstrap
such as `sudo`, `pkexec dpkg <user-file>`, or an unrestricted installer script.
That simply moves the original vulnerability.

Therefore the initial safe release has a packaging prerequisite:

- Distribute the helper through a configured signed apt repository, distro
  package, or explicit administrator-installed signed package.
- The AppImage performs a read-only preflight for the helper path, owner, mode,
  protocol version, and polkit availability.
- If missing, Recommended shows a blocking `privileged_helper_missing` result
  and manual installation instructions before replacing an existing install.
- If authorization is unavailable or denied, Recommended fails closed before
  core mutation because the virtual camera is required.
- Custom may proceed without virtual-camera integration only after the user
  explicitly deselects it and acknowledges the degraded result.

PackageKit could later provide a safe repository-mediated bootstrap, but it
should not be assumed or added without its own threat-model and supported-host
contract.

## Privileged operation protocol v1

The helper should read one bounded JSON request from stdin (or receive the
equivalent typed D-Bus call) and reject all unknown fields. It should accept no
positional operation arguments beyond a protocol switch such as `--json-stdin`.

Example envelope:

```json
{
  "schema_version": 1,
  "request_id": "0f52b85e-6ac0-4dc3-9d38-68ef972bcb7f",
  "transaction_id": "4dd020bc-39aa-42da-b123-44cb2323dca4",
  "plan_digest": "sha256:<64 lowercase hex characters>",
  "policy_version": 1,
  "preconditions": {
    "base_os": "ubuntu",
    "base_release": "24.04",
    "kernel_release": "6.8.0-57-generic"
  },
  "operations": []
}
```

Limits should include a small maximum request size, maximum operation count,
duplicate operation-ID rejection, UTF-8 validation, and exact integer/string
ranges. Canonical plan digest validation remains required in the unprivileged
executor; the helper also validates digest syntax and binds it into results.
The helper's security does not depend on trusting the caller's digest.

### Allowed operations

#### `packages.ensure.v1`

Arguments:

```json
{
  "id": "host-packages",
  "type": "packages.ensure.v1",
  "packages": ["cmake", "ninja-build", "v4l-utils"]
}
```

Rules:

- Only supported Ubuntu bases pass.
- Every package must be an exact member of a compiled policy allowlist.
- The two kernel-derived forms may be accepted only when their suffix exactly
  equals the helper's own `uname` result:
  `linux-headers-<running-kernel>` and
  `linux-modules-extra-<running-kernel>` (plus an explicitly audited
  `linux-modules-<running-kernel>` if supported policy requires it).
- Reject apt flags, package version expressions, repository/source arguments,
  paths, whitespace, control characters, and duplicates.
- Invoke fixed absolute package-manager paths with a sanitized environment and
  `--` before package names. No shell.
- The request must list the exact missing packages shown in review. The helper
  may refresh package metadata only as an explicit documented substep of this
  operation.
- There is no general package removal, purge, autoremove, dpkg-repair, or
  repository-modification operation. Shared dependencies are observed, not
  owned. Broken package state becomes a diagnostic/manual blocker.

The allowlist can contain the currently supported build/runtime, v4l tools,
Vulkan loader, Mesa, and developer shader packages, but policy decides which
are selected. It must not infer Mesa for NVIDIA or shader tools for Recommended.

#### `v4l.module.load.v1`

Arguments:

```json
{
  "id": "load-virtual-camera",
  "type": "v4l.module.load.v1",
  "device_number": 10,
  "label": "StudioCast Camera",
  "exclusive_caps": true
}
```

Rules:

- Module name is compiled as `v4l2loopback`; it is not an argument.
- `device_number` is an integer in the supported range and must have passed the
  analyzed conflict precondition immediately before execution.
- `label` is 1-64 characters from a conservative documented set (letters,
  digits, spaces, `.`, `_`, `-`, `+`, `(`, `)`). Reject quotes, backslashes,
  newlines, other control characters, and invalid UTF-8 rather than attempting
  shell/config escaping.
- `exclusive_caps` is a JSON boolean.
- The helper passes fixed `devices=1`, `video_nr=...`, `card_label=...`, and
  `exclusive_caps=0|1` arguments directly to an absolute `modprobe` executable.
  No shell expansion.

#### `v4l.module.reload.v1`

Uses the same arguments and adds explicit preconditions that the module may be
unloaded and no conflicting/busy device was observed. Unload/reload must never
be an implicit side effect of `load`. A changed/busy precondition returns
`precondition_changed` without unloading anything.

#### `v4l.persistence.write.v1`

Uses the validated module arguments above. It accepts no path and no arbitrary
content. The helper generates canonical content and atomically writes only:

```text
/etc/modules-load.d/studiocast-v4l2loopback.conf
/etc/modprobe.d/studiocast-v4l2loopback.conf
```

It must reject symlinks and non-regular existing files, use safe directory-relative
opens (`openat2`/`openat` with no-follow checks or equivalent), preserve no
unexpected existing content, set root ownership and 0644 mode, fsync before
rename, and return the final SHA-256 hashes for manifest v2.

#### `v4l.persistence.remove.v1`

Arguments contain expected SHA-256 values keyed by the two fixed logical file
IDs, not paths. The helper removes a file only when it is a regular root-owned
file at the compiled path, has the StudioCast marker/schema, and matches the
expected manifest hash. Missing files are idempotent success. Owner, type,
symlink, marker, or hash mismatch returns a typed failure and leaves the file
untouched.

### Explicitly forbidden fields and behavior

Reject a request containing any equivalent of:

```text
command, executable, argv, shell, script, script_path, source_path,
archive_path, build_path, destination_path, cwd, environment, url,
download, file_content, module_name, package_manager_flags
```

The helper must not:

- run `/bin/sh`, `bash -c`, `system()`, or caller-selected executables;
- execute or source any file from the StudioCast source/build/payload tree;
- unpack a caller-selected or downloaded archive as root;
- write a caller-selected path or caller-supplied configuration body;
- inherit caller `PATH`, loader variables, language-runtime variables, or
  plugin paths;
- accept unknown operations for forward compatibility;
- turn a failed sub-operation into overall success.

User-local payload activation, service wiring, models, configuration, cache,
logs, and downloads do not belong in this helper. ONNX Runtime or other
verified archives needed by the application should be staged into the
versioned user-local payload without privilege rather than unpacked into
`/opt`.

## Structured authorization and results

The GUI/client must never parse a password prompt. It should map polkit/client
and helper outcomes into stable reason codes.

Minimum authorization codes:

```text
authorization_granted
authorization_unavailable
authorization_denied
authorization_cancelled
authorization_timeout
privileged_helper_missing
privileged_helper_untrusted
privileged_helper_incompatible
```

Minimum validation/execution codes:

```text
malformed_request
unknown_schema
unknown_operation
unknown_field
unsupported_os
invalid_package
invalid_argument
unsafe_path_state
ownership_mismatch
hash_mismatch
precondition_changed
operation_failed
operation_cancelled
operation_timeout
partial_failure
```

Example result:

```json
{
  "schema_version": 1,
  "request_id": "0f52b85e-6ac0-4dc3-9d38-68ef972bcb7f",
  "transaction_id": "4dd020bc-39aa-42da-b123-44cb2323dca4",
  "plan_digest": "sha256:...",
  "helper_version": "1.0.0",
  "authorization": {"status": "granted", "reason_code": "authorization_granted"},
  "state": "failed",
  "results": [
    {
      "id": "persist-virtual-camera",
      "type": "v4l.persistence.write.v1",
      "status": "failed",
      "changed": false,
      "reason_code": "precondition_changed",
      "exit_code": 3,
      "message": "Virtual-camera device number is now occupied.",
      "files": []
    }
  ]
}
```

Use bounded timeouts. On GUI cancellation, cancel the authorization request or
helper process, close its input, and reap it. Record any operation already
reported as changed in the transaction journal; never manufacture an all-or-
nothing result after a partial host mutation.

There is no silent authorization fallback. Required privilege failure blocks
Recommended before payload activation. Optional Custom host integration may be
retried or explicitly continued in degraded state.

## Packaged uninstall contract

Ordinary uninstall is user-local and must not need the selected source, source
archive, build cache, network, or privilege helper. Prefer implementing the
fixed removal operations directly in the installed backend. Staging a dedicated
uninstall program beside the backend is also acceptable if all installed
callers use that installed path.

The user-level implementation must:

- derive XDG roots from the current user's environment;
- stop/disable only `studiocastd.service` through `systemctl --user` when a
  usable user manager exists;
- remove only fixed StudioCast-owned links, versioned payload roots recorded
  and validated by manifest v2, the StudioCast user unit, desktop entries, and
  runtime artifacts;
- preserve settings/models/data by default;
- reject unsafe, relative, traversal, root, non-owned, and stale manifest
  deletion paths;
- tolerate absent and partial installations;
- record a tombstone/incomplete cleanup result without requiring source files.

Removal of owned host v4l persistence is a separate privileged operation using
`v4l.persistence.remove.v1`. If the trusted helper is unavailable, user-local
uninstall can complete but must report system cleanup as incomplete with manual
instructions; it must not invoke legacy greedy sudo/package purge behavior.

## Required regression tests

### AppDir and uninstall

1. Stage the actual CMake `Installer` component into a temporary AppDir. Assert
   the installed backend's uninstall implementation is present or built in.
2. Extract/use that AppDir layout with temporary `HOME` and all XDG roots plus a
   fake `PATH` containing harmless `systemctl`. Do not give it a source tree.
3. Create only fixture StudioCast links, unit, runtime directory, payload, and
   manifests. Run packaged `status`, `plan uninstall`, and `uninstall --yes`.
4. Assert every reviewed operation is the one executed, exit is zero for an
   absent/healthy/partial fixture, owned artifacts are removed, defaults are
   preserved, and no path outside the fixture changes.
5. Assert the plan contains no missing `${AppDir}/usr/share/scripts/...` path
   and the AppDir tarball verifier requires the same installed removal surface.
6. Add traversal, absolute stale path, directory symlink, target symlink, wrong
   owner/mode where feasible, corrupt manifest, and absent service cases.
7. Remove the fixture build cache before uninstall and prove the active payload
   and uninstaller remain usable.

This test would have caught the current defect because the first packaged
uninstall execution returns 127 in the real staged path shape.

### Privilege client/helper

All tests must use a parser plus fake command/auth/file adapters; no test should
invoke real `sudo`, `pkexec`, apt, modprobe, systemctl, or host `/etc` paths.

Required cases:

- authorization granted, unavailable, denied, cancelled, and timed out;
- missing, wrong-version, user-owned, group/world-writable, and symlink helper;
- malformed/oversized JSON, unknown schema/op/field, duplicates, and bad digest;
- command/script/source/archive/path/env/cwd/URL fields rejected;
- exact package allowlist, unsupported OS, apt flags, version expressions, and
  mismatched kernel package suffixes;
- v4l device range, conflicts, busy reload, boolean caps, empty/long/control/
  newline/quote/backslash labels;
- fixed namespaced persistence output, atomic replacement, symlink rejection,
  ownership/hash checks, idempotent removal, and legacy non-namespaced file
  preservation;
- selected source containing a sentinel `setup.sh` that must never run for a
  privileged operation;
- exact reviewed privileged operations equal fake-helper received operations;
- cancellation reaps the helper and preserves structured partial results;
- no password prompt parser or `sudo` fallback exists in installer execution.

Packaging tests should also verify that the helper package owns the exact
root-helper and polkit paths with expected modes. The AppDir test should verify
only the unprivileged client/contract is bundled, not claim that an AppDir
helper is trusted.

## Implementation acceptance checklist

- AppDir uninstall works with no extracted source and no build cache.
- User-selected source and archives are never executed with privilege and never
  receive authorization credentials.
- The GUI never asks for or transports a sudo password.
- Privileged operations are represented as exact plan operation IDs and typed
  arguments, not prose or rendered shell commands.
- Helper discovery verifies the fixed root-owned artifact and protocol version.
- Only the four operation families above are accepted; additions require a
  schema/policy review.
- Package changes are exact and allowlisted; no purge/autoremove/shared-package
  ownership is introduced.
- v4l files are StudioCast-namespaced, atomically written, and hash-recorded.
- Denial/unavailability/timeouts fail closed and remain distinguishable.
- Unsupported hosts cannot perform automatic package/module changes.
- Recommended detects privilege/v4l blockers before replacing an existing
  installation.
- Tests are hermetic and never exercise real host authorization or mutation.

## Assumptions and caveats

- This audit freezes the privilege contract, not the complete plan/manifest v2
  schemas owned by the backend architecture work.
- Exact package allowlist contents remain tied to the supported Ubuntu release
  policy and should be generated/reviewed with that policy rather than accepted
  from release metadata.
- A root-owned polkit helper package is a real release/install prerequisite. It
  is intentionally not hidden behind optimistic AppImage wording.
- Existing CLI setup scripts may remain documented developer/manual tools, but
  the Recommended GUI/backend path must stop delegating privileged work to them.
- Broad dependency purge and the legacy greedy uninstall are outside the safe
  product uninstall contract and should not be reachable from the GUI.
