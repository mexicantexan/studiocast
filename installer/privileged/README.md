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

The unprivileged installer must first verify the fixed helper path with the
contract in `client_contract.py`, then use polkit and map authorization outcomes
without parsing password prompts. A missing helper blocks Recommended host
integration. Custom may omit the virtual camera only through its explicit
degraded route.
