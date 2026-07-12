#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/VERSION")"
ARCH="$(uname -m)"
BUILD_TYPE="Release"
CMAKE_BUILD_DIR="${REPO_ROOT}/build/appimage-installer"
DIST_DIR="${REPO_ROOT}/dist/appimage"
APPDIR="${DIST_DIR}/StudioCast-Installer-${VERSION}-${ARCH}.AppDir"
BUNDLE_BASENAME="StudioCast-Installer-${VERSION}-${ARCH}"
APPIMAGE_PATH="${DIST_DIR}/${BUNDLE_BASENAME}.AppImage"
APPDIR_TARBALL="${DIST_DIR}/${BUNDLE_BASENAME}.AppDir.tar.gz"
SOURCE_ARCHIVE_NAME="StudioCast-${VERSION}-source.tar.gz"
SOURCE_ARCHIVE_PATH="${DIST_DIR}/${SOURCE_ARCHIVE_NAME}"
STAGED_SOURCE_ARCHIVE="${APPDIR}/usr/share/studiocast/source/${SOURCE_ARCHIVE_NAME}"
CHECKSUM_FILE="${DIST_DIR}/${BUNDLE_BASENAME}.sha256"
LINUXDEPLOY_PATH="${LINUXDEPLOY:-}"
DRY_RUN=0
APPIMAGE_REQUIRED=0
SKIP_APPIMAGE=0
CLEAN=0
APPDIR_EXPLICIT=0
CMAKE_ARGS=(
  -DSTUDIOCAST_ENABLE_OPEN_CUDA=OFF
  -DSTUDIOCAST_ENABLE_OPEN_VULKAN=OFF
  -DSTUDIOCAST_ENABLE_OPEN_AUDIO=OFF
  -DSTUDIOCAST_ENABLE_DLIB=OFF
  -DSTUDIOCAST_BUILD_BENCHMARKS=OFF
)

usage() {
  cat <<EOF
Build a release bundle for the StudioCast GUI installer.

Usage:
  packaging/appimage/build_appimage.sh [options]

Options:
  --help                    Show this help.
  --dry-run                 Print commands without executing them.
  --build-dir DIR           CMake build directory.
                            Default: ${CMAKE_BUILD_DIR}
  --dist-dir DIR            Artifact output directory.
                            Default: ${DIST_DIR}
  --appdir DIR              Staging AppDir path.
                            Default: ${APPDIR}
  --build-type TYPE         CMake build type. Default: Release.
  --cmake-arg ARG           Extra CMake configure argument. May be repeated.
  --linuxdeploy PATH        linuxdeploy executable/AppImage path.
                            Defaults to \$LINUXDEPLOY or PATH lookup.
  --skip-appimage           Only stage and archive the AppDir.
  --appimage-required       Fail if linuxdeploy cannot produce an AppImage.
  --clean                   Remove the CMake build dir and previous artifacts first.

Artifacts:
  ${APPIMAGE_PATH}
  ${APPDIR_TARBALL}
  ${SOURCE_ARCHIVE_PATH}
  ${CHECKSUM_FILE}

Notes:
  The script creates ${SOURCE_ARCHIVE_NAME} with git archive from HEAD when
  possible, stages it into the AppDir at usr/share/studiocast/source/, and also
  leaves the standalone archive in the artifact directory.

  Local builds do not download packaging tools. To produce a self-contained
  AppImage, install linuxdeploy and linuxdeploy-plugin-qt, or pass
  --linuxdeploy. CI downloads those tools explicitly before calling this script.
EOF
}

log() {
  printf '[appimage] %s\n' "$*" >&2
}

die() {
  printf '[appimage] ERROR: %s\n' "$*" >&2
  exit 2
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_cmd "$@"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

write_file() {
  local path="$1"
  local mode="$2"
  shift 2
  log "Writing ${path}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_cmd install -m "${mode}" /dev/stdin "${path}"
    return 0
  fi
  install -d -m 0755 "$(dirname "${path}")"
  {
    printf '%s\n' "$@"
  } > "${path}"
  chmod "${mode}" "${path}"
}

find_linuxdeploy() {
  if [[ -n "${LINUXDEPLOY_PATH}" ]]; then
    [[ -x "${LINUXDEPLOY_PATH}" ]] ||
      die "linuxdeploy is not executable: ${LINUXDEPLOY_PATH}"
    printf '%s\n' "${LINUXDEPLOY_PATH}"
    return 0
  fi

  local candidate
  for candidate in linuxdeploy linuxdeploy-x86_64.AppImage; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done

  return 1
}

find_qmake6() {
  local candidates=()
  if [[ -n "${QMAKE:-}" ]]; then
    candidates+=("${QMAKE}")
  fi
  candidates+=(
    /usr/lib/qt6/bin/qmake6
    /usr/lib/qt6/bin/qmake
    qmake6
    qmake
  )

  local candidate resolved qt_version
  for candidate in "${candidates[@]}"; do
    resolved=""
    if [[ -x "${candidate}" ]]; then
      resolved="${candidate}"
    elif command -v "${candidate}" >/dev/null 2>&1; then
      resolved="$(command -v "${candidate}")"
    fi

    if [[ -n "${resolved}" ]] &&
        qt_version="$("${resolved}" -query QT_VERSION 2>/dev/null)" &&
        [[ "${qt_version}" == 6.* ]]; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done

  return 1
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h)
        usage
        exit 0
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --build-dir)
        [[ $# -ge 2 ]] || die "--build-dir requires a directory"
        CMAKE_BUILD_DIR="$2"
        shift 2
        ;;
      --dist-dir)
        [[ $# -ge 2 ]] || die "--dist-dir requires a directory"
        DIST_DIR="$2"
        shift 2
        ;;
      --appdir)
        [[ $# -ge 2 ]] || die "--appdir requires a directory"
        APPDIR="$2"
        APPDIR_EXPLICIT=1
        shift 2
        ;;
      --build-type)
        [[ $# -ge 2 ]] || die "--build-type requires a value"
        BUILD_TYPE="$2"
        shift 2
        ;;
      --cmake-arg)
        [[ $# -ge 2 ]] || die "--cmake-arg requires a value"
        CMAKE_ARGS+=("$2")
        shift 2
        ;;
      --linuxdeploy)
        [[ $# -ge 2 ]] || die "--linuxdeploy requires a path"
        LINUXDEPLOY_PATH="$2"
        shift 2
        ;;
      --skip-appimage)
        SKIP_APPIMAGE=1
        shift
        ;;
      --appimage-required)
        APPIMAGE_REQUIRED=1
        shift
        ;;
      --clean)
        CLEAN=1
        shift
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

refresh_artifact_paths() {
  BUNDLE_BASENAME="StudioCast-Installer-${VERSION}-${ARCH}"
  if [[ "${APPDIR_EXPLICIT}" -eq 0 ]]; then
    APPDIR="${DIST_DIR}/${BUNDLE_BASENAME}.AppDir"
  fi
  APPIMAGE_PATH="${DIST_DIR}/${BUNDLE_BASENAME}.AppImage"
  APPDIR_TARBALL="${DIST_DIR}/${BUNDLE_BASENAME}.AppDir.tar.gz"
  SOURCE_ARCHIVE_NAME="StudioCast-${VERSION}-source.tar.gz"
  SOURCE_ARCHIVE_PATH="${DIST_DIR}/${SOURCE_ARCHIVE_NAME}"
  STAGED_SOURCE_ARCHIVE="${APPDIR}/usr/share/studiocast/source/${SOURCE_ARCHIVE_NAME}"
  CHECKSUM_FILE="${DIST_DIR}/${BUNDLE_BASENAME}.sha256"
}

configure_and_build() {
  log "Configuring installer build"
  run cmake -S "${REPO_ROOT}" -B "${CMAKE_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    "${CMAKE_ARGS[@]}"

  log "Building studiocast-installer"
  run cmake --build "${CMAKE_BUILD_DIR}" --target studiocast-installer
}

create_source_archive() {
  log "Creating source archive ${SOURCE_ARCHIVE_PATH}"
  run rm -f -- "${SOURCE_ARCHIVE_PATH}"

  if command -v git >/dev/null 2>&1 &&
      git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
      git -C "${REPO_ROOT}" rev-parse --verify HEAD^{commit} >/dev/null 2>&1; then
    run git -C "${REPO_ROOT}" archive \
      --format=tar.gz \
      --prefix="StudioCast-${VERSION}/" \
      --output="${SOURCE_ARCHIVE_PATH}" \
      HEAD
  else
    log "git archive is unavailable; falling back to a working-tree tarball"
    local parent_dir repo_dir
    parent_dir="$(dirname "${REPO_ROOT}")"
    repo_dir="$(basename "${REPO_ROOT}")"
    run tar -C "${parent_dir}" \
      --exclude="${repo_dir}/.git" \
      --exclude="${repo_dir}/build" \
      --exclude="${repo_dir}/dist" \
      --exclude="${repo_dir}/cmake-build-*" \
      -czf "${SOURCE_ARCHIVE_PATH}" \
      --transform "s#^${repo_dir}#StudioCast-${VERSION}#" \
      "${repo_dir}"
  fi

  if [[ "${DRY_RUN}" -eq 0 && ! -f "${SOURCE_ARCHIVE_PATH}" ]]; then
    die "source archive was not created: ${SOURCE_ARCHIVE_PATH}"
  fi
}

stage_appdir() {
  log "Staging AppDir at ${APPDIR}"
  run rm -rf -- "${APPDIR}"
  run cmake --install "${CMAKE_BUILD_DIR}" --prefix "${APPDIR}/usr" --component Installer

  local desktop_dir="${APPDIR}/usr/share/applications"
  local icon_dir="${APPDIR}/usr/share/icons/hicolor/scalable/apps"
  local source_dir="${APPDIR}/usr/share/studiocast/source"
  local desktop_path="${desktop_dir}/studiocast-installer.desktop"
  local icon_path="${icon_dir}/studiocast-installer.svg"

  run install -d -m 0755 "${desktop_dir}" "${icon_dir}" "${source_dir}"
  run install -m 0644 "${REPO_ROOT}/VERSION" "${APPDIR}/usr/share/VERSION"
  run install -m 0644 "${SOURCE_ARCHIVE_PATH}" "${STAGED_SOURCE_ARCHIVE}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_cmd sed "s/@VERSION@/${VERSION}/g" \
      "${SCRIPT_DIR}/studiocast-installer.desktop.in" ">" "${desktop_path}"
  else
    sed "s/@VERSION@/${VERSION}/g" \
      "${SCRIPT_DIR}/studiocast-installer.desktop.in" > "${desktop_path}"
    chmod 0644 "${desktop_path}"
  fi

  run install -m 0644 "${SCRIPT_DIR}/studiocast-installer.svg" "${icon_path}"
  run install -m 0644 "${desktop_path}" "${APPDIR}/studiocast-installer.desktop"
  run install -m 0644 "${icon_path}" "${APPDIR}/studiocast-installer.svg"
  run install -m 0644 "${icon_path}" "${APPDIR}/.DirIcon"

  write_file "${APPDIR}/AppRun" 0755 \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' \
    'export PATH="${HERE}/usr/bin:${PATH}"' \
    'exec "${HERE}/usr/bin/studiocast-installer" "$@"'

  if [[ "${DRY_RUN}" -eq 0 && ! -x "${APPDIR}/usr/bin/studiocast-installer" ]]; then
    die "staged installer binary is missing: ${APPDIR}/usr/bin/studiocast-installer"
  fi
  if [[ "${DRY_RUN}" -eq 0 &&
        ! -x "${APPDIR}/usr/share/studiocast/installer/studiocast-installer-backend" ]]; then
    die "staged installer backend is missing: ${APPDIR}/usr/share/studiocast/installer/studiocast-installer-backend"
  fi
  if [[ "${DRY_RUN}" -eq 0 &&
        ! -f "${APPDIR}/usr/share/studiocast/installer/release/release_channel.py" ]]; then
    die "staged signed-release verifier is missing"
  fi
  if [[ "${DRY_RUN}" -eq 0 &&
        ! -f "${APPDIR}/usr/share/studiocast/installer/release/release-manifest-v1.schema.json" ]]; then
    die "staged release manifest schema is missing"
  fi
  if [[ "${DRY_RUN}" -eq 0 &&
        ! -f "${APPDIR}/usr/share/studiocast/installer/trust/keys/README.md" ]]; then
    die "staged production trust-root contract is missing"
  fi
  if [[ "${DRY_RUN}" -eq 0 && ! -f "${STAGED_SOURCE_ARCHIVE}" ]]; then
    die "staged source archive is missing: ${STAGED_SOURCE_ARCHIVE}"
  fi
}

build_appimage() {
  [[ "${SKIP_APPIMAGE}" -eq 0 ]] || {
    log "Skipping AppImage generation by request"
    return 1
  }

  local linuxdeploy
  if ! linuxdeploy="$(find_linuxdeploy)"; then
    if [[ "${APPIMAGE_REQUIRED}" -eq 1 ]]; then
      die "linuxdeploy was not found; cannot produce required AppImage"
    fi
    log "linuxdeploy not found; leaving staged AppDir archive as the local artifact"
    return 1
  fi

  log "Running linuxdeploy from ${linuxdeploy}"
  local qmake6 qmake_shim_dir
  if ! qmake6="$(find_qmake6)"; then
    if [[ "${APPIMAGE_REQUIRED}" -eq 1 ]]; then
      die "Qt 6 qmake was not found; linuxdeploy-plugin-qt requires qmake -query"
    fi
    log "Qt 6 qmake not found; leaving staged AppDir archive as the local artifact"
    return 1
  fi

  qmake_shim_dir="${CMAKE_BUILD_DIR}/appimage-tools/bin"
  log "Using Qt qmake from ${qmake6}"
  run install -d -m 0755 "${qmake_shim_dir}"
  run ln -sf "${qmake6}" "${qmake_shim_dir}/qmake"

  run rm -f -- "${APPIMAGE_PATH}"
  if ! run env \
    VERSION="${VERSION}" \
    OUTPUT="${APPIMAGE_PATH}" \
    APPIMAGE_EXTRACT_AND_RUN="${APPIMAGE_EXTRACT_AND_RUN:-1}" \
    QMAKE="${qmake6}" \
    PATH="${qmake_shim_dir}:${PATH}" \
    "${linuxdeploy}" \
    --appdir "${APPDIR}" \
    --executable "${APPDIR}/usr/bin/studiocast-installer" \
    --desktop-file "${APPDIR}/usr/share/applications/studiocast-installer.desktop" \
    --icon-file "${APPDIR}/usr/share/icons/hicolor/scalable/apps/studiocast-installer.svg" \
    --plugin qt \
    --output appimage; then
    if [[ "${APPIMAGE_REQUIRED}" -eq 1 ]]; then
      die "linuxdeploy failed to produce the AppImage"
    fi
    log "linuxdeploy failed; leaving staged AppDir archive as the local artifact"
    return 1
  fi

  if [[ "${DRY_RUN}" -eq 0 && ! -f "${APPIMAGE_PATH}" ]]; then
    local generated
    generated="$(find "${REPO_ROOT}" "${DIST_DIR}" -maxdepth 1 -type f -name '*.AppImage' -print | head -n 1 || true)"
    if [[ -n "${generated}" ]]; then
      run mv -f -- "${generated}" "${APPIMAGE_PATH}"
    fi
  fi

  if [[ "${DRY_RUN}" -eq 0 && ! -f "${APPIMAGE_PATH}" ]]; then
    if [[ "${APPIMAGE_REQUIRED}" -eq 1 ]]; then
      die "expected AppImage was not created: ${APPIMAGE_PATH}"
    fi
    log "expected AppImage was not created; leaving staged AppDir archive as the local artifact"
    return 1
  fi

  run chmod +x "${APPIMAGE_PATH}"
  return 0
}

archive_appdir() {
  log "Archiving AppDir"
  run rm -f -- "${APPDIR_TARBALL}"
  run tar -C "$(dirname "${APPDIR}")" -czf "${APPDIR_TARBALL}" "$(basename "${APPDIR}")"
}

write_checksums() {
  log "Writing checksums"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_cmd sha256sum "${APPIMAGE_PATH}" "${APPDIR_TARBALL}" \
      "${SOURCE_ARCHIVE_PATH}" ">" "${CHECKSUM_FILE}"
    return 0
  fi

  : > "${CHECKSUM_FILE}"
  local artifact
  for artifact in "${APPIMAGE_PATH}" "${APPDIR_TARBALL}" "${SOURCE_ARCHIVE_PATH}"; do
    if [[ -f "${artifact}" ]]; then
      (cd "${DIST_DIR}" && sha256sum "$(basename "${artifact}")") >> "${CHECKSUM_FILE}"
    fi
  done
}

main() {
  parse_args "$@"
  refresh_artifact_paths

  if [[ "${CLEAN}" -eq 1 ]]; then
    log "Cleaning previous build and artifacts"
    run rm -rf -- "${CMAKE_BUILD_DIR}" "${APPDIR}" "${APPIMAGE_PATH}" \
      "${APPDIR_TARBALL}" "${SOURCE_ARCHIVE_PATH}" "${CHECKSUM_FILE}"
  fi

  run install -d -m 0755 "${DIST_DIR}"
  configure_and_build
  create_source_archive
  stage_appdir
  build_appimage || true
  archive_appdir
  write_checksums

  log "Artifacts:"
  [[ "${DRY_RUN}" -eq 1 || -f "${APPIMAGE_PATH}" ]] && log "  ${APPIMAGE_PATH}"
  log "  ${APPDIR_TARBALL}"
  log "  ${SOURCE_ARCHIVE_PATH}"
  log "  ${CHECKSUM_FILE}"
}

main "$@"
