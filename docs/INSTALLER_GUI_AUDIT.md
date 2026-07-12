# StudioCast installer GUI audit and frozen routing contract

Date: 2026-07-12

Audited baseline: `v0.2.9` plus the Wave 1 backend and Vulkan audit contracts

Scope: `installer/gui/installer_wizard.*`, `installer/gui/main.cpp`, and
`tests/installer_wizard_tests.cpp`. This document changes no production code.

## Policy inputs

This contract applies the repository policies in `docs/GUI_POLICY.md`,
`docs/MODEL_POLICY.md`, `docs/INSTALLER_BACKEND_AUDIT.md`, and
`docs/INSTALLER_VULKAN_CAPABILITIES.md`:

- The GUI is a controller and presentation layer. Backend facts,
  recommendations, plans, execution results, and manifest health are
  authoritative.
- Recommended installation is user-local, uses only a pinned and verified
  official source release, requires virtual-camera integration, and selects the
  seven default model packs even when Maxine is usable.
- Arbitrary source trees/archives, custom build/cache paths, developer package
  controls, and unsupported overrides are Advanced/developer concerns.
- Host integration is independently authorized through typed operations from a
  trusted packaged helper. The GUI must not offer broad `sudo`, execute a
  selected source as root, or imply that running the whole installer as root is
  acceptable.
- Vulkan is evaluated per effect. For this release it is not automatically
  recommended for any effect; its useful paths remain degraded, experimental,
  diagnostics-only, or unavailable as recorded in the Vulkan audit.
- Missing optional features remain visible. A partial optional failure must not
  erase a committed core installation or become full success.

## Executive finding

The present wizard is a flag editor around two independent backend calls, not a
state-driven transaction UI. It cannot prove that what the user reviewed is
what ran, and it cannot preserve an execution failure through the Finish page.
The current short uninstall route is worth retaining, but its plan validation
and result handling have the same defects as the long route.

The replacement must be driven by this data flow:

```text
analyze -> facts -> recommend(facts, intent, prior desired config, policy)
        -> selections -> exact reviewed plan + digest/token
        -> execute that plan -> structured events + terminal result
        -> re-analyze for reconciliation, without replacing the result
```

No GUI workflow name or widget default may stand in for facts, prior desired
configuration, a reviewed plan, or an execution result.

## Evidence-backed defects

### 1. Routing ignores detected state

Confirmed.

- `IntroPage` permanently presents six radio buttons: install, update, repair,
  uninstall, clean install, and advanced
  (`installer/gui/installer_wizard.cpp:675-718`). Install is always initially
  checked.
- The displayed installation state is only the old `installed` boolean and a
  version string (`731-765`). There is no routing for partial, unhealthy,
  stale, tombstone, unknown manifest schema, installed-newer, offline, or
  cache-only states.
- Invalid combinations are consequently ordinary choices: update or repair on
  an absent installation, install over an unhealthy installation, and a normal
  downgrade are not state-gated.

### 2. Analysis failure is misclassified as unsupported OS

Confirmed.

- Any status/backend error creates a synthetic `installed=false` result and an
  OS object with `supported=false` (`507-526`). A missing backend, timeout, or
  malformed JSON therefore looks like an unsupported distribution.
- The compatibility page exposes “Allow this workflow on an unsupported
  distro” for every such state and makes the page complete when checked
  (`791-850`). The flag is then passed to the ordinary workflow (`604-605`).
- This fails closed neither for analysis failure nor unsupported package/module
  changes. Unsupported override belongs only in a CLI/developer surface.

### 3. Workflow names overwrite installed desired configuration

Confirmed.

- Constructor state hard-codes dependencies, v4l, module loading/persistence,
  service, and open backends on; models, Vulkan, Mesa, and shader tools off
  (`installer/gui/installer_wizard.h:94-114`). Source and cache paths are also
  locally derived (`installer_wizard.cpp:335-346`).
- Entering Dependency Plan resets dependency, v4l, load, persistence, service,
  and fresh-build choices solely from the workflow string (`863-874`). Update
  and repair therefore lose prior choices before review.
- No GUI state consumes manifest-v2 `desired_configuration`, represents an
  unknown migrated field, or distinguishes desired from actual service/v4l
  state.

### 4. Plan-generation failure does not block Apply

Confirmed.

- Dependency Plan, Uninstall, and Review replace their text with an error when
  planning fails (`876-881`, `961-968`, `1351-1359`). None implements a
  completeness or validation condition tied to a valid plan.
- `refreshPlan` clears `planObject_`, but navigation and commit remain enabled
  (`529-539`). Progress then starts a newly reconstructed workflow anyway.
- Blockers, warnings, schema/policy compatibility, expiry, digest, token, and
  changed preconditions are not represented by the GUI.

### 5. The reviewed plan is not the execution input

Confirmed.

- Review calls `plan <workflow> --json` with options assembled from widget
  state (`529-539`).
- Progress later calls the workflow name with another freshly assembled option
  list and `--yes` (`541-623`, `1428-1433`). It neither persists nor passes the
  reviewed plan object, digest, or approval token.
- Current tests validate argument fragments rather than equality between
  reviewed and executed operation IDs. A test explicitly expects the defective
  Finish commit label (`tests/installer_wizard_tests.cpp:114-130`).

### 6. “Finish” initiates mutation

Confirmed.

- The wizard and Review page label their commit button “Finish”
  (`installer_wizard.cpp:352`, `1351-1354`). Advancing initializes Progress,
  which immediately starts the backend (`1375-1433`).
- “Finish” therefore means “begin changes,” contrary to the ordinary meaning
  of the label and the required action-specific Apply labels.
- The actual terminal page is also generically titled Finish, so the same word
  means both commit and close.

### 7. Failed execution can become an apparent successful completion

Confirmed.

- Any normal failure, crash, or start failure sets `complete_=true`
  (`1411-1439`). `validatePage` always returns true after refreshing status
  (`1443-1447`), so failure advances through the same route as success.
- `exitCode_` is private to Progress and is not consumed by Finish. Exit kind,
  signal, failed operation, transaction state, rollback, core commitment, and
  retryable operations are discarded.
- Finish refreshes status and infers its message only from `installed`
  (`1468-1507`). A failed update with the old version still installed becomes
  “StudioCast is installed”; an optional model failure may become “not
  installed”; an uninstall against a corrupt status may become “was removed.”
- There is no representation of `degraded`, `rolled_back`, or `cancelled`, and
  no Retry/Continue behavior for optional failures.

### 8. Cancellation does not supervise the transaction or descendants

Confirmed.

- `ProgressPage` owns a `QProcess`, but there is no wizard `reject`, window
  close, process-error, cancellation request, or bounded termination handler
  (`installer_wizard.h:228-243`).
- Closing the wizard can at most destroy/kill the direct `QProcess`; the GUI
  does not request backend transaction cancellation, wait for compensation, or
  prove that child processes are gone.
- No cancellation result or journal is retained, and the GUI cannot prevent a
  late manifest commit after the window disappears.

### 9. Offline, unsupported, degraded, and trust states are absent

Confirmed.

- Connectivity and verified release/model cache availability are not UI state.
  Work may begin before an unavailable artifact is discovered.
- Unsupported systems can opt into the same automatic setup UI instead of
  being limited to diagnostics/manual instructions.
- Optional component notices are derived from presence-shaped legacy status,
  not per-effect usability. A Vulkan loader notice is especially insufficient
  given the audited production capability matrix.
- Ordinary install/update pages expose source directories and arbitrary
  archives. Recommended is therefore not constrained to verified official
  source, and the UI does not explain the privileged helper trust boundary.

## Frozen GUI state-routing contract

### Authority and state model

The GUI consumes the frozen backend `installer-facts/v1`,
`installer-recommendation/v1`, `installer-plan/v1`, and structured execution
result. It validates the schema/version envelope before rendering. Unknown
schemas produce **Analysis unavailable**, never “not installed” or
“unsupported.”

The analysis model kept by the GUI contains:

- the exact facts document and `host_fingerprint`;
- installation classification and manifest health;
- active/previous/target versions and version relation;
- prior desired configuration, including explicit `unknown` migrated values;
- connectivity and verified cache facts;
- the selected route, intent, and recommendation document;
- a dirty selection draft;
- at most one current plan, its exact serialized file, digest, token, and
  expiry;
- at most one execution transaction and terminal result.

Changing any selection, route, preservation choice, source, target version, or
advanced field invalidates and deletes the pending GUI plan reference. Apply is
disabled until the backend creates a new valid plan from the current draft.

### Analyze-page action matrix

The first page analyzes automatically and shows one primary action derived from
facts. It is not a workflow radio list.

| Detected facts | Primary action | Secondary/contextual actions |
| --- | --- | --- |
| `absent`, supported, required release reachable or verified cached | **Install recommended** | Customize |
| `healthy`, target newer | **Update recommended** | Customize, Repair, Reinstall/reset, Uninstall |
| `healthy`, same version | **Modify installation** | Repair, Reinstall/reset, Uninstall |
| `healthy`, installed newer than channel | **Keep current** (no mutation) | Customize, Repair, Reinstall/reset, Uninstall; downgrade only in Advanced |
| `partial` or `unhealthy` with safe ownership/recovery facts | **Repair recommended** | Customize, Reinstall/reset when safe, Uninstall when safe |
| `stale` | **Reconstruct/repair** when ownership can be proven; otherwise **Review safe cleanup** | Diagnostics; no unchecked deletion |
| `tombstone` with no owned residue | **Install recommended** | Customize |
| `tombstone` with validated residue | **Review safe cleanup** | Install only after reconciliation |
| corrupt/truncated/unknown manifest schema | **Review diagnostics** | Safe reconstruction only; destructive actions blocked until ownership is proven |
| unsupported OS/architecture | **View diagnostics and manual instructions** | No Recommended; Custom is diagnostic/manual only; no automatic package/module changes |
| offline with every required artifact in verified cache | State-appropriate action above, labeled **Use verified cache** in its summary | Show artifact ages/identities |
| offline with a required core artifact absent/unverified | **Retry analysis** | Manual download/cache instructions; no mutation |
| analyzer/backend/protocol failure | **Retry analysis** | Copy diagnostics, manual-download fallback when applicable |

“Keep current” never creates or executes a plan. A self-update recommendation is
a separate banner/action: download and verify the newer installer, then **Restart
with new installer** or choose manual download. It never replaces a running
AppImage silently.

Contextual actions are shown only when facts make them meaningful and safe:

- **Customize** enters the Custom route seeded from recommendation and prior
  desired configuration.
- **Repair** reconciles the same installed version and prior desired choices.
- **Reinstall/reset** implements clean install and starts with one **Preserve
  settings** checkbox checked. It preserves only user configuration; payload,
  service wiring, cache, logs/runtime state, and models are reset.
- **Uninstall** retains the short confirmation route. It preserves settings,
  models, and data by default and removes only validated StudioCast-owned
  artifacts.

### Route contracts

Recommended route:

```text
Analyze -> Recommended setup and review -> Progress -> Complete
```

The Recommended page is backend-authored and read-only except for entering
Customize. It shows the selected version, official verified source identity,
Release build, user-local destination, service decision, required virtual
camera, all seven default pack IDs, per-effect engine choices/reasons, download
sizes/licenses, blockers, warnings, and exact categorized operations. Any
virtual-camera blocker, unsupported OS, low disk, missing unverified core
artifact, or invalid plan blocks the action before mutation.

Custom route:

```text
Analyze
  -> Scope/application
  -> Features
  -> System integration
  -> Models/downloads
  -> Preservation
  -> Review/validation
  -> Progress
  -> Complete
```

- **Scope/application** always says User-local; there is no system-wide
  choice. Official verified source remains the normal choice.
- **Features** shows product effects and deterministic per-effect engine
  availability/reasons. Maxine, CUDA, Vulkan, and CPU are not globally
  inferred. Vulkan developer choices carry the audit limitations.
- **System integration** separates application scope from trusted-helper work.
  A usable systemd user manager defaults service to enabled and started.
  Virtual camera defaults on. Turning it off requires a prominent
  acknowledgment that StudioCast cannot be selected as a camera and that the
  final state will be degraded.
- **Models/downloads** selects the exact seven defaults initially, even with
  Maxine. Custom may deselect individual packs and sees pack IDs, eight artifact
  files, sizes, hashes/signatures, provenance, license, cache, and network
  state. Existing custom packs are preserved during update/repair.
- **Preservation** appears when relevant. Reinstall has only **Preserve
  settings**, checked by default. Unknown migrated preservation or ownership
  facts are explicit blockers, not defaults.
- **Review/validation** is the only mutation commit page.

Advanced is an explicit disclosure within Customize, not a sixth ordinary
workflow. It may expose source directory/archive, build type, build/cache path,
fresh build, individual providers/backends, Vulkan/Mesa/shader tools, model IDs
and destination, v4l device number/label/caps, and downgrade selection.
Developer source is labeled unverified and may never supply privileged code.
Unsupported override instructions are CLI/developer-only. System-wide scope is
not exposed.

The short uninstall route remains:

```text
Analyze -> Uninstall review -> Progress -> Complete
```

Its review must list every owned path/integration operation, every preserved
data class, and every ownership mismatch left untouched. A plan error or unsafe
ownership fact blocks Uninstall. “No manifest-managed install” is not permission
to delete fixed path guesses.

### Selection seeding contract

There are no workflow-name resets in page initialization.

1. Install starts from the pure recommendation.
2. Update, modify, repair, and reinstall start from manifest-v2 prior desired
   configuration, then apply only explicit new policy safety constraints and
   user changes.
3. Repair keeps the same version unless update is separately selected and
   includes only actual-versus-desired differences.
4. Migrated `unknown` fields remain unknown and require review when material.
5. Page revisits render the selection draft without rewriting it.
6. Recommended always uses verified official release source, Release build,
   version-specific cache, required v4l, usable systemd service by default, and
   seven default packs.

### Review and Apply contract

Review renders the exact `installer-plan/v1`, grouped as:

- Application files and versioned payload activation/rollback;
- Downloads, with URL/source, size, hash/signature, provenance, and license;
- User configuration and preservation;
- Service changes;
- Virtual-camera/system changes;
- Privileged trusted-helper operations;
- Preserved items;
- Removed items;
- Unavailable optional features and degraded outcomes;
- Expected validation and reversible/irreversible status.

Operation IDs and dependencies are available under details. Human review text
must be generated from operation objects, never a second list of guessed
commands. Blockers are visually distinct from warnings.

The commit button is enabled only when the plan is nonempty, schema/policy
supported, blocker-free, unexpired, and still matches the current selections
and facts fingerprint. Its exact label is:

| Plan intent | Commit label |
| --- | --- |
| install | **Install** |
| update or modify desired configuration | **Update** |
| repair | **Repair** |
| reinstall/reset | **Reinstall** |
| uninstall | **Uninstall** |

“Finish” is reserved for closing a terminal outcome and never initiates work.

On commit, the GUI persists the exact backend-produced plan in a private file
and invokes only:

```text
execute-plan --plan <file> --digest <reviewed-digest> --token <reviewed-token>
```

It does not reconstruct feature/workflow flags. Executor rejection for digest,
token, expiry, schema, facts, precondition, or operation mismatch returns to
Review with a blocker and requires re-analysis/re-planning.

### Structured progress contract

Progress is driven by backend operation events keyed by transaction ID, plan
ID/digest, operation ID, phase, state, and structured error. Recommended phase
groups are:

1. Preflight and authorization
2. Acquire and verify source
3. Build
4. Stage and validate payload
5. Virtual camera/system integration
6. Activate payload and links
7. Reconcile service
8. Install/verify models
9. Validate and record manifest

The primary view shows the active phase, completed/total operations, current
operation, and plain-language status. The exact log remains available behind
**Show details** and is retained for support. Raw subprocess output cannot mark
an operation or transaction successful.

The GUI requires a parseable terminal execution result bound to the reviewed
plan. Process exit code/status corroborates the result; it does not replace it.
A zero exit without a valid terminal result, a mismatched plan/transaction, a
crash, or EOF in `prepared`, `executing`, or `core_committed` state is a
protocol failure and never full success.

Back navigation is disabled once execution starts. Closing or Cancel during
execution asks for confirmation, sends the backend cancellation request/signal,
and waits asynchronously for a structured `cancelled`/rollback result. The
backend owns process-group termination and compensation. A bounded timeout may
terminate an unresponsive direct executor, but the GUI must then report an
uncertain interrupted transaction, re-analyze, and never claim children were
cleaned up or a manifest was committed without evidence.

### Completion/result contract

The terminal page is based first on the structured execution result. A fresh
analysis is shown as reconciliation evidence only and cannot turn failure into
success.

| Terminal state | User-visible outcome |
| --- | --- |
| `committed`, core and overall healthy | **Installation/Update/Repair/Reinstall/Uninstall complete**; Finish closes |
| `degraded`, `core_committed=true` | **Complete with limitations**; list active version, failed optional operations and effects; offer **Retry** and **Continue** |
| `failed`, no new core committed | **Action failed**; preserve and name the prior active payload when present; offer retry/review diagnostics |
| `rolled_back` | **Update failed; previous version restored**; show validation/error and active prior version |
| `cancelled` | **Action cancelled**; show compensation and actual active state; never show complete |
| authorization denied/cancelled/timeout | Explicit authorization result and unchanged operations; no sudo fallback |
| malformed/missing result or interrupted unknown state | **Installer interrupted; status needs reconciliation**; run analysis and offer Repair/diagnostics |

Retry for optional failures requests a new backend plan limited to current
retryable differences; it never reuses a single-use token. Continue accepts the
accurate degraded state and closes. Recommended cannot report committed success
without its required virtual camera. Custom without virtual camera always ends
degraded, even when every selected operation succeeded.

## Required GUI tests

Use fixture facts, recommendations, plans, event streams, and fake backend
processes under temporary HOME/XDG roots. No test may use network, sudo, pkexec,
package managers, module/service mutation, or real model downloads.

At minimum add coverage for:

- every Analyze matrix row: absent, healthy/current, update available,
  installed newer, partial, unhealthy, stale, tombstone, corrupt/truncated,
  unknown schema, unsupported, offline cache-complete/incomplete, and analyzer
  failure;
- Recommended and Custom page sequences, contextual action visibility, no
  system-wide option, Advanced-only source/downgrade controls, and retained
  short uninstall route;
- update/repair/reinstall seeding from prior desired service, v4l, backend,
  build, model, and preservation choices; page revisit must not reset values;
- exact seven default model IDs/eight artifacts, Maxine-present fallbacks, and
  no automatically recommended Vulkan effect in this release;
- plan error/blocker/expiry/digest/facts mismatch disabling commit; any edit
  invalidating the current plan;
- exact action labels and a regression assertion that **Finish** cannot begin
  mutation;
- execution invocation containing only the reviewed plan file/digest/token,
  with reviewed operation IDs equal to executed IDs plus declared
  compensation;
- structured success, build failure, model-degraded core success, rollback,
  cancellation, authorization denial/cancellation/timeout, crash, start
  failure, malformed result, and EOF before a terminal state;
- failed update preserving the prior active payload, and failure never becoming
  successful merely because refreshed status says an installation exists;
- Retry/Continue behavior for optional failures and required-v4l Recommended
  blockers;
- cancellation of a fake backend with descendants, bounded UI behavior, no
  child left behind, and no falsely committed manifest;
- unsupported Custom showing diagnostics/manual instructions without automatic
  package/module operations, and offline missing-core state stopping before any
  mutation;
- packaged AppDir backend/uninstall routing and installer self-update mock
  download/verify/relaunch without overwriting the running AppImage.

## Compatibility and implementation notes

- Preserve current backend CLI flags only in compatibility launchers. The new
  GUI path uses facts/recommendation/plan APIs and `execute-plan`.
- Preserve the recently shortened uninstall UX, but replace its guessed cleanup
  and reconstructed command with the same plan/result contracts.
- Preserve Open Video/Open CUDA names and CMake identifiers; the GUI may present
  neutral per-effect capability language without a repository-wide rename.
- `docs/DEVELOPER_GUIDE.md` and `docs/SETUP.md` describe older build-cache links
  and clean-install preservation. The backend audit and redesign requirements
  supersede those passages: active payloads are durable/versioned, and clean
  install preserves only settings when requested.
- No system-wide application installation is part of this contract. A trusted
  root-owned helper/package may be a packaging prerequisite for v4l operations;
  the GUI must report that prerequisite rather than weaken the boundary.
