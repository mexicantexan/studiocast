# StudioCast privileged protocol

`studiocast_system_helper.py` is the source payload for the separately signed
`studiocast-system-helper` OS package. It is not installed from, or trusted from,
the AppImage, selected source tree, archive, build cache, or user data roots.

The helper accepts only `--json-stdin` and the five versioned operations frozen
in `docs/INSTALLER_PRIVILEGE_AUDIT.md`. Requests are bounded and fully validated
before the first operation. Unknown schema versions, operations, and fields are
rejected. Commands use fixed absolute argument arrays and a fixed environment;
there is no shell, arbitrary executable, caller path, caller content, or sudo
fallback.

The unprivileged installer calls `preflight_trusted_helper()` before planning
host integration. This read-only check verifies the fixed helper's ownership,
mode, version, and protocol plus the fixed `pkexec`, policy binding, system bus,
and polkit service descriptor. It never requests authorization and returns a
stable installer-facts object through `HelperPreflightResult.as_dict()`.

Only execution uses polkit and maps authorization outcomes without parsing
password prompts. A missing, incompatible, or transport-unavailable helper
blocks Recommended host integration. Custom may omit the virtual camera only
through its explicit degraded route.
