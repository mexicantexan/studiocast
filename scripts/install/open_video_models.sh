#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRANSACTION="${REPO_ROOT}/installer/models/studiocast-model-transaction"
CATALOG="${REPO_ROOT}/packaging/models/curated-model-catalog-v1.json"
DEFAULT_DEST="${XDG_DATA_HOME:-${HOME}/.local/share}/studiocast/models/open_video"
DEFAULT_CACHE="${XDG_CACHE_HOME:-${HOME}/.cache}/studiocast/models"

usage() {
  cat <<EOF
StudioCast Open Video Model Installer

Usage:
  $0 [--dest <path>] [--model <id>] [--force] [--offline]

Options:
  --dest <path>       Open Video model root (default: ${DEFAULT_DEST})
  --model <id>        Install one curated pack; repeatable.
  --force             Re-stage the selected curated pack.
  --list              List curated Open Video packs and exit.
  --include-placeholders
                      Accepted for CLI compatibility; curated placeholders are
                      never installed by this transaction path.
  --cache <path>      Verified artifact cache (default: ${DEFAULT_CACHE})
  --source <url/path> Configurable HTTP(S), file://, or local artifact root.
  --offline           Use only verified installed artifacts or cache.
  --retries <count>   Retry each source candidate (default: 2).
  --recommended       Require trusted signed artifact sizes before mutation.
  --json              Emit the structured result as JSON.
  -h, --help          Show help.

Legacy STUDIOCAST_HF_REPO_ID and STUDIOCAST_HF_REV remain supported.
STUDIOCAST_MODEL_SOURCE takes precedence. Maxine does not replace or deselect
the five default Open Video fallback packs.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

dest="${DEFAULT_DEST}"
cache="${DEFAULT_CACHE}"
source="${STUDIOCAST_MODEL_SOURCE:-}"
retries="2"
force=0
offline=0
recommended=0
json=0
list=0
include_placeholders=0
models=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) [[ $# -ge 2 ]] || die "--dest requires a path"; dest="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || die "--model requires an id"; models+=("$2"); shift 2 ;;
    --force) force=1; shift ;;
    --list) list=1; shift ;;
    --include-placeholders) include_placeholders=1; shift ;;
    --cache) [[ $# -ge 2 ]] || die "--cache requires a path"; cache="$2"; shift 2 ;;
    --source) [[ $# -ge 2 ]] || die "--source requires a URL or path"; source="$2"; shift 2 ;;
    --offline) offline=1; shift ;;
    --retries) [[ $# -ge 2 ]] || die "--retries requires a count"; retries="$2"; shift 2 ;;
    --recommended) recommended=1; shift ;;
    --json) json=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

if [[ "${include_placeholders}" -eq 1 && "${json}" -ne 1 ]]; then
  echo "WARN: curated placeholder packs are intentionally unavailable." >&2
fi

if [[ "${list}" -eq 1 ]]; then
  [[ "${json}" -eq 1 ]] || echo "Curated Open Video packs:"
  args=(--catalog "${CATALOG}" --repo-root "${REPO_ROOT}" list --family open_video)
  [[ "${json}" -eq 1 ]] && args+=(--json)
  exec "${TRANSACTION}" "${args[@]}"
fi

if [[ -z "${source}" && -n "${STUDIOCAST_HF_REPO_ID:-}" ]]; then
  source="https://huggingface.co/${STUDIOCAST_HF_REPO_ID}/resolve/${STUDIOCAST_HF_REV:-main}"
fi

args=(--catalog "${CATALOG}" --repo-root "${REPO_ROOT}" install --family open_video --dest "${dest}" --cache "${cache}" --retries "${retries}")
[[ -n "${source}" ]] && args+=(--source "${source}")
[[ "${force}" -eq 1 ]] && args+=(--force)
[[ "${offline}" -eq 1 ]] && args+=(--offline)
[[ "${recommended}" -eq 1 ]] && args+=(--recommended)
[[ "${json}" -eq 1 ]] && args+=(--json)
for model in "${models[@]}"; do args+=(--model "${model}"); done

exec "${TRANSACTION}" "${args[@]}"
