#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=packaging/appimage/tools.lock
source "${SCRIPT_DIR}/tools.lock"

VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/VERSION")"
ARCH="$(uname -m)"
DIST_DIR="${REPO_ROOT}/dist/appimage"
APPDIR=""
REQUIRE_APPIMAGE=0
APPIMAGE_RUNTIME_PATH="${APPIMAGE_RUNTIME:-}"
TRUSTED_RELEASE_KEYS=()

usage() {
  cat <<EOF
Verify StudioCast installer packaging artifacts.

Usage:
  packaging/appimage/verify_bundle.sh [options]

Options:
  --dist-dir DIR          Artifact output directory.
                          Default: ${DIST_DIR}
  --appdir DIR            Staged AppDir path. Defaults to the AppDir expected
                          under --dist-dir for VERSION and uname -m.
  --require-appimage      Require and verify the AppImage artifact.
  --appimage-runtime PATH SHA-pinned type-2 runtime used to verify the
                          AppImage executable prefix without running it.
  --trusted-release-key ID=PATH
                          Require this exact public key in the staged AppDir
                          and final AppImage. May be repeated.
  --help                  Show this help.
EOF
}

die() {
  printf '[verify-appimage] ERROR: %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[verify-appimage] %s\n' "$*" >&2
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "missing file: ${path}"
}

require_executable() {
  local path="$1"
  [[ -x "${path}" ]] || die "missing executable: ${path}"
}

require_equal() {
  local actual="$1"
  local expected="$2"
  local context="$3"
  [[ "${actual}" == "${expected}" ]] ||
    die "${context}: expected '${expected}', got '${actual}'"
}

checksum_contains() {
  local checksum_file="$1"
  local artifact="$2"
  grep -F "  $(basename "${artifact}")" "${checksum_file}" >/dev/null ||
    die "checksum file does not reference $(basename "${artifact}")"
}

tarball_contains() {
  local tarball="$1"
  local entry="$2"
  local label="${3:-tarball}"
  tar -tzf "${tarball}" "${entry}" >/dev/null ||
    die "${label} does not contain ${entry}"
}

desktop_value() {
  local path="$1"
  local key="$2"
  awk -F= -v key="${key}" '$1 == key {
    print substr($0, length($1) + 2)
    exit
  }' "${path}"
}

require_desktop_value() {
  local path="$1"
  local key="$2"
  local expected="$3"
  local actual
  actual="$(desktop_value "${path}" "${key}")"
  require_equal "${actual}" "${expected}" "$(basename "${path}") ${key}"
}

capture_command() {
  local description="$1"
  shift
  local output
  if ! output="$("$@" 2>&1)"; then
    die "${description} failed: ${output}"
  fi
  printf '%s' "${output}"
}

require_command_output_contains() {
  local description="$1"
  local expected="$2"
  shift 2
  local output
  output="$(capture_command "${description}" "$@")"
  [[ "${output}" == *"${expected}"* ]] ||
    die "${description} output did not contain '${expected}': ${output}"
}

validate_service_file() {
  local path="$1"
  grep -Fx 'Type=simple' "${path}" >/dev/null ||
    die "service file missing Type=simple: ${path}"
  grep -Fx 'ExecStart=%h/.local/bin/studiocastd' "${path}" >/dev/null ||
    die "service file has unexpected ExecStart: ${path}"
  grep -Fx 'WantedBy=default.target' "${path}" >/dev/null ||
    die "service file missing WantedBy=default.target: ${path}"
}

verify_staged_metadata() {
  local appdir="$1"
  local expected_version="$2"
  local source_archive_path="$3"
  local staged_source_archive="$4"
  local desktop_path="${appdir}/usr/share/applications/studiocast-installer.desktop"
  local root_desktop_path="${appdir}/studiocast-installer.desktop"
  local version_path="${appdir}/usr/share/VERSION"
  local desktop_id="studiocast-installer"

  require_file "${version_path}"
  require_file "${appdir}/AppRun"
  require_file "${appdir}/studiocast-installer.svg"
  require_file \
    "${appdir}/usr/share/icons/hicolor/scalable/apps/studiocast-installer.svg"

  require_equal "$(tr -d '[:space:]' < "${version_path}")" \
    "${expected_version}" "staged VERSION"
  cmp -s "${desktop_path}" "${root_desktop_path}" ||
    die "top-level desktop file differs from usr/share/applications copy"
  cmp -s "${source_archive_path}" "${staged_source_archive}" ||
    die "staged source archive differs from standalone source archive"

  require_desktop_value "${desktop_path}" "Type" "Application"
  require_desktop_value "${desktop_path}" "Name" "StudioCast Installer"
  require_desktop_value "${desktop_path}" "Exec" "${desktop_id}"
  require_desktop_value "${desktop_path}" "Icon" "${desktop_id}"
  require_desktop_value "${desktop_path}" "Terminal" "false"
  require_desktop_value "${desktop_path}" "X-AppImage-Version" \
    "${expected_version}"
  require_equal "$(basename "${desktop_path}")" "${desktop_id}.desktop" \
    "desktop file ID"

  require_command_output_contains "staged installer --version" \
    "studiocast-installer ${expected_version}" \
    "${appdir}/usr/bin/studiocast-installer" --version
  require_command_output_contains "AppRun --version" \
    "studiocast-installer ${expected_version}" \
    "${appdir}/AppRun" --version
  require_command_output_contains "installer backend --help" \
    "usage: studiocast-installer-backend" \
    "${appdir}/usr/share/studiocast/installer/studiocast-installer-backend" \
    --help
  require_file "${appdir}/usr/share/studiocast/installer/release/release_channel.py"
  require_file "${appdir}/usr/share/studiocast/installer/release/release-manifest-v1.schema.json"
  require_file "${appdir}/usr/share/studiocast/installer/privileged/__init__.py"
  require_file "${appdir}/usr/share/studiocast/installer/privileged/client_contract.py"
  require_file "${appdir}/usr/share/studiocast/installer/trust/keys/README.md"
  local model_launcher="${appdir}/usr/share/studiocast/installer/models/studiocast-model-transaction"
  require_executable "${model_launcher}"
  require_file "${appdir}/usr/share/studiocast/installer/models/model_transactions.py"
  require_file "${appdir}/usr/share/studiocast/packaging/models/curated-model-catalog-v1.json"

  local model_summary
  model_summary="$(capture_command "packaged curated model catalog" "${model_launcher}" list --json)"
  python3 - "${model_summary}" <<'PY'
import json
import sys

summary = json.loads(sys.argv[1])
defaults = [pack for pack in summary["packs"] if pack["default"]]
if len(defaults) != 7:
    raise SystemExit(f"packaged catalog has {len(defaults)} default packs, expected 7")
if sum(len(pack["artifacts"]) for pack in defaults) != 8:
    raise SystemExit("packaged default catalog does not contain exactly 8 artifacts")
if summary.get("size_metadata", {}).get("trusted") is not True:
    raise SystemExit("packaged default catalog does not have trusted artifact sizes")
if not summary.get("size_metadata", {}).get("reason_code"):
    raise SystemExit("packaged default catalog has no trusted-size evidence reason")
for pack in defaults:
    for artifact in pack["artifacts"]:
        if (type(artifact.get("size_bytes")) is not int or
                artifact["size_bytes"] <= 0 or
                artifact.get("size_status") != "known"):
            raise SystemExit("packaged default catalog has incomplete trusted artifact sizes")
PY

  local forbidden
  forbidden="$(find "${appdir}/usr/share/studiocast" -type f \
    \( -name '*.onnx' -o -name '*.ort' -o -name '*.engine' -o \
       -name '*.plan' -o -name '*.dat' -o -name '*.bin' \) -print -quit)"
  [[ -z "${forbidden}" ]] ||
    die "AppDir redistributes a model binary: ${forbidden}"

  local service
  while IFS= read -r service; do
    validate_service_file "${service}"
  done < <(find "${appdir}" -type f -name '*.service' -print)
}

verify_trust_roots() {
  local appdir="$1"
  local required="$2"
  local trust_dir="${appdir}/usr/share/studiocast/installer/trust/keys"
  local count=0 key key_id
  while IFS= read -r key; do
    count=$((count + 1))
    [[ ! -L "${key}" ]] || die "release trust root may not be a symlink: ${key}"
    key_id="$(basename "${key}" .pem)"
    [[ "${key_id}" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] ||
      die "invalid release trust-root key ID: ${key_id}"
    [[ ! "${key_id}" =~ (^|[._-])(test|fixture)([._-]|$) ]] ||
      die "test/fixture key was staged in the production trust root: ${key_id}"
    openssl pkey -pubin -in "${key}" -text -noout 2>/dev/null |
      grep -q 'ED25519' || die "release trust root is not Ed25519: ${key}"
  done < <(find "${trust_dir}" -maxdepth 1 -type f -name '*.pem' -print | sort)
  if [[ "${required}" -eq 1 && "${count}" -eq 0 ]]; then
    die "release AppImage contains no production release trust root"
  fi
}

verify_expected_trust_roots() {
  local appdir="$1"
  shift
  local trust_dir="${appdir}/usr/share/studiocast/installer/trust/keys"
  local specification key_id key_path packaged_key
  for specification in "$@"; do
    [[ "${specification}" == *=* ]] ||
      die "--trusted-release-key requires ID=PATH"
    key_id="${specification%%=*}"
    key_path="${specification#*=}"
    [[ "${key_id}" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] ||
      die "invalid expected release key ID: ${key_id}"
    require_file "${key_path}"
    [[ ! -L "${key_path}" ]] ||
      die "expected release public key may not be a symlink: ${key_path}"
    packaged_key="${trust_dir}/${key_id}.pem"
    require_file "${packaged_key}"
    [[ ! -L "${packaged_key}" ]] ||
      die "packaged release trust root may not be a symlink: ${packaged_key}"
    cmp -s "${key_path}" "${packaged_key}" ||
      die "packaged release trust root differs from expected key: ${key_id}"
  done
}

secret_pem_pattern='-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY-----'

verify_no_secret_pem_in_tree() {
  local root="$1"
  local match
  match="$(find "${root}" -type f -exec grep -alE -- "${secret_pem_pattern}" {} + || true)"
  [[ -z "${match}" ]] || die "forbidden secret PEM marker in staged artifact: ${match}"
}

verify_no_secret_pem_in_source_archive() {
  local archive="$1"
  if tar -xOf "${archive}" | grep -aE -- "${secret_pem_pattern}" >/dev/null; then
    die "forbidden secret PEM marker in source archive"
  fi
}

verify_no_secret_pem_in_dist_files() {
  local dist_dir="$1"
  local match
  match="$(find "${dist_dir}" -maxdepth 1 -type f -exec grep -alE -- \
    "${secret_pem_pattern}" {} + || true)"
  [[ -z "${match}" ]] || die "forbidden secret PEM marker in upload artifact: ${match}"
}

verify_final_appimage() {
  local appimage appdir_tar runtime
  appimage="$(realpath -e -- "$1")"
  appdir_tar="$(realpath -e -- "$2")"
  runtime="$(realpath -e -- "$3")"
  shift 3
  local temporary extracted
  temporary="$(mktemp -d)"
  extracted="${temporary}/squashfs-root"
  if ! python3 "${SCRIPT_DIR}/verify_type2_appimage.py" \
      --appimage "${appimage}" \
      --runtime-file "${runtime}" \
      --runtime-sha256 "${APPIMAGE_RUNTIME_SHA256}" \
      --appdir-tar "${appdir_tar}" \
      --extract-dir "${extracted}"; then
    rm -rf -- "${temporary}"
    die "independent final AppImage verification failed"
  fi
  verify_trust_roots "${extracted}" 1
  verify_expected_trust_roots "${extracted}" "$@"
  verify_no_secret_pem_in_tree "${extracted}"
  rm -rf -- "${temporary}"
}

verify_packaged_backend_layout() {
  local appdir="$1"
  local backend="${appdir}/usr/share/studiocast/installer/studiocast-installer-backend"
  local temporary
  temporary="$(mktemp -d)"

  local home="${temporary}/home"
  local data="${temporary}/data"
  local config="${temporary}/config"
  local state="${temporary}/state"
  local cache="${temporary}/cache"
  local fake_bin="${temporary}/bin"
  install -d -m 0755 "${home}/.local/bin" "${data}/studiocast/payloads/fixture/bin" \
    "${data}/studiocast/models/custom" "${config}/studiocast" "${state}" "${cache}" "${fake_bin}"
  cat >"${fake_bin}/systemctl" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod 0755 "${fake_bin}/systemctl"
  cat >"${data}/studiocast/payloads/fixture/bin/studiocast" <<'EOF'
#!/usr/bin/env bash
echo "studiocast fixture"
EOF
  chmod 0755 "${data}/studiocast/payloads/fixture/bin/studiocast"
  ln -s "${data}/studiocast/payloads/fixture" "${data}/studiocast/current"
  ln -s "${data}/studiocast/current/bin/studiocast" "${home}/.local/bin/studiocast"
  printf 'preserve\n' >"${data}/studiocast/models/custom/user-model"
  printf 'preserve\n' >"${config}/studiocast/daemon.conf"

  local -a environment=(env "HOME=${home}" "XDG_DATA_HOME=${data}" "XDG_CONFIG_HOME=${config}"
    "XDG_STATE_HOME=${state}" "XDG_CACHE_HOME=${cache}" "PATH=${fake_bin}:/usr/bin:/bin")
  "${environment[@]}" "${backend}" status --json >/dev/null
  local plan
  plan="$("${environment[@]}" "${backend}" plan uninstall --json --preserve-user-data \
    --no-v4l2loopback --no-service --no-models --allow-unsupported)"
  [[ "${plan}" != *"usr/share/scripts"* ]] ||
    die "packaged uninstall plan refers to a source-tree uninstall script"
  [[ "${plan}" == *'"links.reconcile"'* ]] ||
    die "packaged uninstall plan omits link reconciliation"

  rm -rf -- "${cache}"
  "${data}/studiocast/current/bin/studiocast" >/dev/null ||
    die "active payload broke after build/release cache removal"
  "${environment[@]}" "${backend}" uninstall --yes --preserve-user-data \
    --no-v4l2loopback --no-service --no-models --allow-unsupported >/dev/null
  [[ ! -e "${data}/studiocast/payloads" ]] || die "packaged uninstall left owned payloads"
  [[ ! -L "${home}/.local/bin/studiocast" ]] || die "packaged uninstall left owned links"
  [[ -f "${data}/studiocast/models/custom/user-model" ]] || die "packaged uninstall removed models by default"
  [[ -f "${config}/studiocast/daemon.conf" ]] || die "packaged uninstall removed settings by default"
  rm -rf -- "${temporary}"
}

verify_source_archive_layout() {
  local archive="$1"
  local prefix="$2"
  local expected_version="$3"

  require_equal "${prefix}" "StudioCast-${expected_version}" \
    "source archive top-level directory"

  tarball_contains "${archive}" "${prefix}/VERSION" "source archive"
  tarball_contains "${archive}" "${prefix}/CMakeLists.txt" "source archive"
  tarball_contains "${archive}" \
    "${prefix}/installer/backend/studiocast-installer-backend" \
    "source archive"
  tarball_contains "${archive}" \
    "${prefix}/packaging/appimage/studiocast-installer.desktop.in" \
    "source archive"
  tarball_contains "${archive}" \
    "${prefix}/packaging/systemd/user/studiocastd.service" \
    "source archive"
  tarball_contains "${archive}" \
    "${prefix}/resources/model_packs/open_video/README.md" \
    "source archive"

  local archived_version
  archived_version="$(tar -xOf "${archive}" "${prefix}/VERSION" |
    tr -d '[:space:]')"
  require_equal "${archived_version}" "${expected_version}" \
    "source archive VERSION"

  local entry
  while IFS= read -r entry; do
    case "${entry}" in
      "${prefix}"|"${prefix}/"*) ;;
      *)
        die "source archive contains unexpected top-level entry: ${entry}"
        ;;
    esac
    case "${entry}" in
      "${prefix}/.git"|\
      "${prefix}/.git/"*|\
      "${prefix}/build"|\
      "${prefix}/build/"*|\
      "${prefix}/dist"|\
      "${prefix}/dist/"*|\
      "${prefix}/cmake-build-"*)
        die "source archive contains generated or VCS path: ${entry}"
        ;;
    esac
  done < <(tar -tzf "${archive}")
}

verify_source_archive_metadata() {
  local archive="$1"
  local prefix="$2"
  local expected_version="$3"

  command -v python3 >/dev/null 2>&1 ||
    die "python3 is required for source metadata validation"

  python3 - "${archive}" "${prefix}" "${expected_version}" <<'PY'
import json
import posixpath
import sys
import tarfile

archive, prefix, expected_version = sys.argv[1:4]


def fail(message):
    print(f"[verify-appimage] ERROR: {message}", file=sys.stderr)
    sys.exit(2)


def read_text(tar, name):
    try:
        member = tar.extractfile(name)
    except KeyError:
        fail(f"source archive does not contain {name}")
    if member is None:
        fail(f"source archive member is not a file: {name}")
    return member.read().decode("utf-8")


def is_safe_relative_path(value):
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("/") or "\\" in value:
        return False
    parts = value.split("/")
    return all(part not in ("", ".", "..") for part in parts)


with tarfile.open(archive, "r:*") as tar:
    names = {member.name for member in tar.getmembers()}
    version = read_text(tar, f"{prefix}/VERSION").strip()
    if version != expected_version:
        fail(f"source VERSION mismatch: expected {expected_version}, got {version}")

    desktop_template = read_text(
        tar, f"{prefix}/packaging/appimage/studiocast-installer.desktop.in"
    )
    for required in (
        "Exec=studiocast-installer",
        "Icon=studiocast-installer",
        "X-AppImage-Version=@VERSION@",
    ):
        if required not in desktop_template:
            fail(f"desktop template missing {required}")

    service = read_text(tar, f"{prefix}/packaging/systemd/user/studiocastd.service")
    for required in (
        "Type=simple",
        "ExecStart=%h/.local/bin/studiocastd",
        "WantedBy=default.target",
    ):
        if required not in service:
            fail(f"systemd service missing {required}")

    manifests = sorted(
        name
        for name in names
        if name.startswith(f"{prefix}/resources/model_packs/")
        and name.endswith("/model.json")
    )
    audio_manifests = [
        name
        for name in manifests
        if name.startswith(f"{prefix}/resources/model_packs/open_audio/")
    ]
    video_manifests = [
        name
        for name in manifests
        if name.startswith(f"{prefix}/resources/model_packs/open_video/")
    ]
    if not audio_manifests:
        fail("source archive contains no Open Audio model manifests")
    if not video_manifests:
        fail("source archive contains no Open Video model manifests")

    seen_ids = set()
    for name in manifests:
        data = json.loads(read_text(tar, name))
        model_id = data.get("id")
        if not isinstance(model_id, str) or not model_id:
            fail(f"{name} has an empty model id")
        if model_id in seen_ids:
            fail(f"duplicate model id in source archive: {model_id}")
        seen_ids.add(model_id)

        pack_dir = posixpath.dirname(name)
        if f"{pack_dir}/LICENSE.txt" not in names:
            fail(f"{name} pack is missing LICENSE.txt")

        relative_pack = name.removeprefix(f"{prefix}/resources/model_packs/")
        parts = relative_pack.split("/")
        if parts[0] == "open_audio":
            if len(parts) != 3:
                fail(f"unexpected Open Audio manifest path: {name}")
        elif parts[0] == "open_video":
            if len(parts) != 4:
                fail(f"unexpected Open Video manifest path: {name}")
            task = data.get("task")
            if task != parts[1]:
                fail(f"{name} task '{task}' does not match path '{parts[1]}'")
        else:
            fail(f"unexpected model manifest root: {name}")

        references = []
        onnx_filename = data.get("onnx_filename")
        if onnx_filename is not None:
            references.append(("onnx_filename", onnx_filename))
        files = data.get("files")
        if isinstance(files, list):
            for index, item in enumerate(files):
                if isinstance(item, dict) and "name" in item:
                    references.append((f"files[{index}].name", item["name"]))
        onnx = data.get("onnx")
        if isinstance(onnx, dict):
            if "main" in onnx:
                references.append(("onnx.main", onnx["main"]))
            models = onnx.get("models")
            if isinstance(models, list):
                for index, item in enumerate(models):
                    if isinstance(item, dict) and "file" in item:
                        references.append((f"onnx.models[{index}].file", item["file"]))

        if not references:
            fail(f"{name} does not declare any model file paths")
        for key, value in references:
            if not is_safe_relative_path(value):
                fail(f"{name} has unsafe {key}: {value!r}")
PY
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dist-dir)
        [[ $# -ge 2 ]] || die "--dist-dir requires a directory"
        DIST_DIR="$2"
        shift 2
        ;;
      --appdir)
        [[ $# -ge 2 ]] || die "--appdir requires a directory"
        APPDIR="$2"
        shift 2
        ;;
      --require-appimage)
        REQUIRE_APPIMAGE=1
        shift
        ;;
      --appimage-runtime)
        [[ $# -ge 2 ]] || die "--appimage-runtime requires a path"
        APPIMAGE_RUNTIME_PATH="$2"
        shift 2
        ;;
      --trusted-release-key)
        [[ $# -ge 2 ]] || die "--trusted-release-key requires ID=PATH"
        TRUSTED_RELEASE_KEYS+=("$2")
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  local bundle_basename="StudioCast-Installer-${VERSION}-${ARCH}"
  local appimage_path="${DIST_DIR}/${bundle_basename}.AppImage"
  local appdir_tarball="${DIST_DIR}/${bundle_basename}.AppDir.tar.gz"
  local source_archive_name="StudioCast-${VERSION}-source.tar.gz"
  local source_archive_path="${DIST_DIR}/${source_archive_name}"
  local checksum_file="${DIST_DIR}/${bundle_basename}.sha256"

  if [[ -z "${APPDIR}" ]]; then
    APPDIR="${DIST_DIR}/${bundle_basename}.AppDir"
  fi

  local staged_source_archive
  staged_source_archive="${APPDIR}/usr/share/studiocast/source/${source_archive_name}"
  local appdir_base
  appdir_base="$(basename "${APPDIR}")"

  require_file "${appdir_tarball}"
  require_file "${source_archive_path}"
  require_file "${checksum_file}"
  require_executable "${APPDIR}/usr/bin/studiocast-installer"
  require_executable "${APPDIR}/usr/share/studiocast/installer/studiocast-installer-backend"
  require_file "${APPDIR}/usr/share/applications/studiocast-installer.desktop"
  require_file "${APPDIR}/studiocast-installer.desktop"
  require_file "${staged_source_archive}"

  local source_prefix
  source_prefix="StudioCast-${VERSION}"

  if [[ "${REQUIRE_APPIMAGE}" -eq 1 ]]; then
    require_executable "${appimage_path}"
    require_file "${APPIMAGE_RUNTIME_PATH}"
    checksum_contains "${checksum_file}" "${appimage_path}"
  elif [[ -f "${appimage_path}" ]]; then
    require_file "${APPIMAGE_RUNTIME_PATH}"
    checksum_contains "${checksum_file}" "${appimage_path}"
  fi

  checksum_contains "${checksum_file}" "${appdir_tarball}"
  checksum_contains "${checksum_file}" "${source_archive_path}"
  (cd "${DIST_DIR}" && sha256sum --check "$(basename "${checksum_file}")")

  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/bin/studiocast-installer" "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/studiocast/installer/studiocast-installer-backend" \
    "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/studiocast/installer/release/release_channel.py" \
    "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/studiocast/installer/trust/keys/README.md" \
    "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/studiocast/installer/models/studiocast-model-transaction" \
    "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/studiocast/packaging/models/curated-model-catalog-v1.json" \
    "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/applications/studiocast-installer.desktop" \
    "AppDir tarball"
  tarball_contains "${appdir_tarball}" \
    "${appdir_base}/usr/share/studiocast/source/${source_archive_name}" \
    "AppDir tarball"

  verify_staged_metadata "${APPDIR}" "${VERSION}" "${source_archive_path}" \
    "${staged_source_archive}"
  verify_trust_roots "${APPDIR}" "${REQUIRE_APPIMAGE}"
  verify_expected_trust_roots "${APPDIR}" "${TRUSTED_RELEASE_KEYS[@]}"
  verify_no_secret_pem_in_tree "${APPDIR}"
  verify_no_secret_pem_in_source_archive "${source_archive_path}"
  verify_no_secret_pem_in_dist_files "${DIST_DIR}"
  if [[ -f "${appimage_path}" ]]; then
    verify_final_appimage "${appimage_path}" "${appdir_tarball}" \
      "${APPIMAGE_RUNTIME_PATH}" "${TRUSTED_RELEASE_KEYS[@]}"
  fi
  verify_packaged_backend_layout "${APPDIR}"
  verify_source_archive_layout "${source_archive_path}" "${source_prefix}" \
    "${VERSION}"
  verify_source_archive_metadata "${source_archive_path}" "${source_prefix}" \
    "${VERSION}"

  if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate \
      "${APPDIR}/usr/share/applications/studiocast-installer.desktop"
  else
    log "desktop-file-validate not found; skipping desktop metadata validation."
  fi

  log "Packaging artifact checks passed for ${bundle_basename}."
}

main "$@"
