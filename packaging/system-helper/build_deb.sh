#!/usr/bin/env bash
set -euo pipefail

# Builds an unsigned .deb payload without privilege or network access. Release
# infrastructure must sign and distribute it through a trusted package channel.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${1:-${REPO_ROOT}/dist/system-helper}"
VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/VERSION")"
ARCH="all"
STAGE="$(mktemp -d)"
trap 'rm -rf -- "${STAGE}"' EXIT
chmod 0755 "${STAGE}"

command -v dpkg-deb >/dev/null 2>&1 || {
  echo "build_deb.sh: dpkg-deb is required" >&2
  exit 1
}

install -d -m 0755 \
  "${STAGE}/DEBIAN" \
  "${STAGE}/usr/libexec/studiocast" \
  "${STAGE}/usr/share/polkit-1/actions"
install -m 0755 \
  "${REPO_ROOT}/installer/privileged/studiocast_system_helper.py" \
  "${STAGE}/usr/libexec/studiocast/studiocast-system-helper"
install -m 0644 \
  "${SCRIPT_DIR}/com.studiocast.Installer1.policy" \
  "${STAGE}/usr/share/polkit-1/actions/com.studiocast.Installer1.policy"

printf '%s\n' \
  'Package: studiocast-system-helper' \
  "Version: ${VERSION}-1" \
  "Architecture: ${ARCH}" \
  'Maintainer: StudioCast maintainers' \
  'Depends: python3, policykit-1' \
  'Section: admin' \
  'Priority: optional' \
  'Description: Trusted allowlisted host-integration helper for StudioCast' \
  ' This package supplies the root-owned polkit execution boundary. It does not' \
  ' install StudioCast application payloads or bootstrap itself from an AppImage.' \
  > "${STAGE}/DEBIAN/control"
chmod 0644 "${STAGE}/DEBIAN/control"

mkdir -p -- "${OUTPUT_DIR}"
OUTPUT="${OUTPUT_DIR}/studiocast-system-helper_${VERSION}-1_${ARCH}.deb"
dpkg-deb --root-owner-group --build "${STAGE}" "${OUTPUT}"
echo "${OUTPUT}"
