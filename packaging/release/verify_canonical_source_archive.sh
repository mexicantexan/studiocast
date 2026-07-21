#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: verify_canonical_source_archive.sh ARCHIVE COMMIT VERSION [REPOSITORY]

Regenerate the exact release source archive with trusted Git tooling and compare
its bytes with ARCHIVE. REPOSITORY defaults to the current working directory.
EOF
}

die() {
  printf '[verify-source-archive] ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -ge 3 && $# -le 4 ]] || {
  usage >&2
  exit 2
}

archive="$1"
commit="$2"
version="$3"
repository="${4:-.}"

[[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  die "VERSION must be strict MAJOR.MINOR.PATCH"
[[ -f "${archive}" && ! -L "${archive}" ]] ||
  die "archive must be a regular non-symlink file: ${archive}"
git -C "${repository}" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
  die "repository is not a Git worktree: ${repository}"
resolved_commit="$(git -C "${repository}" rev-parse --verify "${commit}^{commit}")" ||
  die "commit is unavailable: ${commit}"
[[ "$(git -C "${repository}" show "${resolved_commit}:VERSION" | tr -d '[:space:]')" == "${version}" ]] ||
  die "commit VERSION does not match ${version}"

temporary="$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/studiocast-source.XXXXXX.tar.gz")"
cleanup() {
  rm -f -- "${temporary}"
}
trap cleanup EXIT

git -C "${repository}" archive \
  --format=tar.gz \
  --prefix="StudioCast-${version}/" \
  --output="${temporary}" \
  "${resolved_commit}"

cmp -s "${temporary}" "${archive}" ||
  die "source archive is not the canonical git archive for ${resolved_commit}"
printf '%s\n' "${resolved_commit}"
