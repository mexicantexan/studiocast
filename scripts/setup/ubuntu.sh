#!/usr/bin/env bash
set -euo pipefail

# StudioCast Ubuntu-family setup helper.
#
# This script is invoked via ./scripts/setup.sh.
# It installs build/runtime prerequisites and configures v4l2loopback.

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup.sh [options]

Options:
  --deps                 Install build/runtime deps (Qt/CMake/Ninja/etc + Pulse utils).
  --v4l2loopback          Ensure v4l2loopback module is available (kernel module or DKMS).
  --load-loopback         Load v4l2loopback now (creates /dev/videoN).
  --persist-loopback      Persist module load/options across reboot.

  --video-nr N            v4l2loopback device number (default: 10).
  --label TEXT            v4l2loopback card label (default: "StudioCast Camera").
  --exclusive-caps 0|1    v4l2loopback exclusive_caps (default: 1).

  --onnxruntime-version V ONNX Runtime version to install (default: 1.17.3).
  --onnxruntime-flavor    cpu|gpu (default: auto; gpu if nvidia-smi works, else cpu).
  --onnxruntime-arch A    x64|aarch64 (default: auto from uname -m).
  --vulkan-runtime        Install optional Vulkan loader/diagnostic packages.
  --mesa-vulkan           With --vulkan-runtime, install Mesa Intel/AMD ICDs.
  --shader-tools          Install optional developer shader tools.

  --build                 Configure + build StudioCast (dev convenience).
  --build-dir DIR         Build directory (default: ./cmake-build-debug).
  --build-type TYPE       CMake build type (default: Debug).

  --maxine                Run Maxine helper (see scripts/setup/maxine.sh).
  -y, --yes               Assume yes for apt installs.
  -h, --help              Show help.

Examples:
  ./scripts/setup.sh --deps --v4l2loopback --load-loopback --persist-loopback
  ./scripts/setup.sh --deps --onnxruntime-flavor gpu
EOF
}

YES=0
DO_DEPS=0
DO_V4L2=0
DO_LOAD_LOOP=0
DO_PERSIST_LOOP=0
VIDEO_NR=10
LABEL="StudioCast Camera"
EXCLUSIVE_CAPS=1
DO_BUILD=0
BUILD_DIR="./cmake-build-debug"
BUILD_TYPE="Debug"
DO_MAXINE=0
DO_VULKAN_RUNTIME=0
DO_MESA_VULKAN=0
DO_SHADER_TOOLS=0
MAXINE_ARGS=()
PARSE_MAXINE_ARGS=0

# ONNX Runtime install defaults.
# - Prefer GPU flavor if an NVIDIA driver is present.
# - Let users override via CLI flags.
ORT_VERSION="${ORT_VERSION:-1.17.3}"

if [[ -z "${ORT_ARCH:-}" ]]; then
  case "$(uname -m)" in
    x86_64|amd64) ORT_ARCH="x64" ;;
    aarch64|arm64) ORT_ARCH="aarch64" ;;
    *) ORT_ARCH="x64" ;;
  esac
fi

if [[ -z "${ORT_FLAVOR:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ORT_FLAVOR="gpu"
  else
    ORT_FLAVOR="cpu"
  fi
fi

log() { echo "[setup] $*"; }

if [[ "${STUDIOCAST_GUI_SUDO_STDIN:-0}" == "1" ]]; then
  sudo() {
    command sudo -S -p "${STUDIOCAST_GUI_SUDO_PROMPT:-[sudo] password for %u: }" "$@"
  }
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[setup] Missing required command: $1"; exit 1; }
}

onnxruntime_package_name() {
  local arch="$1"
  local flavor="$2"
  local version="$3"

  if [[ "${flavor}" == "cpu" ]]; then
    printf 'onnxruntime-linux-%s-%s\n' "${arch}" "${version}"
  else
    printf 'onnxruntime-linux-%s-%s-%s\n' "${arch}" "${flavor}" "${version}"
  fi
}

APT_ARGS=()
APT_GET_ARGS=()

apt_install() {
  local pkgs=("$@")
  sudo apt update
  sudo apt install "${APT_ARGS[@]}" "${pkgs[@]}"
}

ensure_onnxruntime_available() {
  require_cmd curl
  require_cmd pkg-config
  require_cmd tar

  if pkg-config --exists onnxruntime; then
    log "onnxruntime already available via pkg-config; skipping ONNX Runtime install."
    return 0
  fi

  local ort_name
  ort_name="$(onnxruntime_package_name "${ORT_ARCH}" "${ORT_FLAVOR}" "${ORT_VERSION}")"
  local ort_tgz="${ort_name}.tgz"
  local ort_url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ort_tgz}"

  # Install location for the extracted upstream tarball.
  local ort_prefix="/opt/studiocast/onnxruntime/${ORT_VERSION}"
  local ort_root="${ort_prefix}/${ort_name}"

  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT

  log "Installing ONNX Runtime ${ORT_VERSION} (${ORT_FLAVOR}) from: ${ort_url}"
  log "  -> ${ort_root}"
  if [[ "${ORT_FLAVOR}" == "gpu" ]]; then
    log "Note: ORT flavor 'gpu' requires a working NVIDIA driver/CUDA runtime stack at runtime."
  fi

  curl -fsSL "${ort_url}" -o "${tmpdir}/${ort_tgz}"

  sudo mkdir -p "${ort_prefix}"
  sudo tar -xzf "${tmpdir}/${ort_tgz}" -C "${ort_prefix}"

  if [[ ! -f "${ort_root}/include/onnxruntime_cxx_api.h" ]]; then
    echo "[setup] ERROR: ONNX Runtime headers not found at ${ort_root}/include/onnxruntime_cxx_api.h"
    exit 1
  fi

  local ort_libdir="${ort_root}/lib"
  if [[ -d "${ort_root}/lib64" ]]; then
    ort_libdir="${ort_root}/lib64"
  fi

  if [[ ! -e "${ort_libdir}/libonnxruntime.so" ]]; then
    local sofile
    sofile="$(ls -1 "${ort_libdir}"/libonnxruntime.so.* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${sofile}" ]]; then
      echo "[setup] ERROR: libonnxruntime.so not found under ${ort_libdir}"
      exit 1
    fi
    sudo ln -sf "$(basename "${sofile}")" "${ort_libdir}/libonnxruntime.so"
  fi

  # Make the shared library discoverable for runtime linking.
  sudo tee /etc/ld.so.conf.d/studiocast-onnxruntime.conf >/dev/null <<EOF
${ort_libdir}
EOF
  sudo ldconfig

  # Provide a pkg-config file so our CMake can pick it up via pkg_check_modules(onnxruntime).
  sudo mkdir -p /usr/local/lib/pkgconfig
  sudo tee /usr/local/lib/pkgconfig/onnxruntime.pc >/dev/null <<EOF
prefix=${ort_root}
exec_prefix=\${prefix}
libdir=${ort_libdir}
includedir=\${prefix}/include

Name: onnxruntime
Description: ONNX Runtime
Version: ${ORT_VERSION}
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
EOF

  log "ONNX Runtime installed; pkg-config reports: $(pkg-config --modversion onnxruntime)"
}

have_module() {
  # Does the module exist for this running kernel?
  modinfo v4l2loopback >/dev/null 2>&1
}

ensure_kernel_extras() {
  # On some installs the module may live in linux-modules-extra for the current kernel.
  # Installing it is safe even if already present.
  apt_install "linux-modules-extra-$(uname -r)" || true
}

pkg_status() {
  # Print dpkg status keyword for a package, or empty if unknown.
  # Examples: install ok installed, install ok half-configured, deinstall ok config-files, etc.
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null || true
}

is_pkg_installedish() {
  # True if package is installed OR half-installed/half-configured/unpacked (i.e., can break apt).
  local st
  st="$(pkg_status "$1")"
  [[ "$st" == install\ ok\ installed ]] || \
  [[ "$st" == install\ ok\ unpacked ]] || \
  [[ "$st" == install\ ok\ half-configured ]] || \
  [[ "$st" == install\ ok\ half-installed ]] || \
  [[ "$st" == install\ ok\ triggers-awaited ]] || \
  [[ "$st" == install\ ok\ triggers-pending ]]
}

fix_broken_dkms_if_newer_kernel_module_exists() {
  # If v4l2loopback exists in the kernel AND v4l2loopback-dkms is installed-ish,
  # purge it to avoid dpkg repeatedly failing on future apt installs.
  if have_module; then
    if is_pkg_installedish v4l2loopback-dkms; then
      log "Detected in-kernel v4l2loopback + v4l2loopback-dkms in dpkg state: '$(pkg_status v4l2loopback-dkms)'."
      log "Purging v4l2loopback-dkms (and dkms) to avoid version conflicts..."

      # Use apt-get here for more predictable noninteractive behavior.
      sudo apt-get update
      sudo apt-get remove --purge "${APT_GET_ARGS[@]}" v4l2loopback-dkms dkms || true
      sudo apt-get -f install "${APT_GET_ARGS[@]}" || true
      sudo dpkg --configure -a || true

      # If something still lingers, try one more time.
      if is_pkg_installedish v4l2loopback-dkms; then
        log "v4l2loopback-dkms still present after purge attempt; trying again..."
        sudo apt-get remove --purge "${APT_GET_ARGS[@]}" v4l2loopback-dkms dkms || true
        sudo apt-get -f install "${APT_GET_ARGS[@]}" || true
        sudo dpkg --configure -a || true
      fi

      if is_pkg_installedish v4l2loopback-dkms; then
        echo "[setup] ERROR: Could not remove v4l2loopback-dkms cleanly."
        echo "[setup] Please run manually:"
        echo "  sudo apt remove --purge v4l2loopback-dkms dkms"
        echo "  sudo apt -f install"
        echo "  sudo dpkg --configure -a"
        exit 1
      fi

      log "DKMS conflict cleaned up."
    fi
  fi
}

ensure_v4l2loopback_available() {
  log "Ensuring v4l2loopback availability..."

  # If kernel has it, don't touch DKMS.
  if have_module; then
    log "v4l2loopback module is available for this kernel (no DKMS needed)."
    # Ensure runtime tools are present; modules-extra is harmless and may be required on minimal installs.
    apt_install "linux-modules-extra-$(uname -r)" v4l2loopback-utils v4l-utils || true
    return 0
  fi

  # Try installing modules-extra for this kernel, then re-check.
  ensure_kernel_extras
  if have_module; then
    log "v4l2loopback became available after installing linux-modules-extra."
    apt_install v4l2loopback-utils v4l-utils || true
    return 0
  fi

  # DKMS fallback only if not present.
  log "v4l2loopback not found in kernel modules; installing DKMS fallback."
  apt_install dkms "linux-headers-$(uname -r)" v4l2loopback-dkms v4l2loopback-utils v4l-utils

  if have_module; then
    log "v4l2loopback is now available (DKMS)."
  else
    echo "[setup] ERROR: v4l2loopback still not available after DKMS install."
    echo "[setup] Check dkms status: dkms status"
    exit 1
  fi
}

install_optional_vulkan_runtime() {
  log "Installing optional Vulkan runtime/diagnostic packages..."
  local pkgs=(libvulkan1 vulkan-tools)
  if [[ "${DO_MESA_VULKAN}" -eq 1 ]]; then
    pkgs+=(mesa-vulkan-drivers)
  fi
  apt_install "${pkgs[@]}"
  log "Vulkan packages only provide a loader/diagnostic path. Open Vulkan also requires a build with -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON and a GPU driver/ICD exposing a compute-capable Vulkan device."
  if [[ "${DO_MESA_VULKAN}" -ne 1 ]]; then
    log "For Intel/AMD Mesa drivers, rerun with --vulkan-runtime --mesa-vulkan or install mesa-vulkan-drivers manually. NVIDIA Vulkan support normally comes from the NVIDIA driver packages."
  fi
}

install_optional_shader_tools() {
  log "Installing optional developer shader tools..."
  apt_install glslang-tools
  log "Shader tools are only needed to regenerate/validate embedded Vulkan SPIR-V headers; normal builds use committed generated headers."
}

load_v4l2loopback_now() {
  require_cmd modprobe
  log "Loading v4l2loopback now (video_nr=${VIDEO_NR}, label=${LABEL}, exclusive_caps=${EXCLUSIVE_CAPS})..."
  sudo modprobe -r v4l2loopback 2>/dev/null || true
  sudo modprobe v4l2loopback "video_nr=${VIDEO_NR}" "card_label=${LABEL}" "exclusive_caps=${EXCLUSIVE_CAPS}"
  log "Loaded. Devices:"
  if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --list-devices || true
  fi
  ls -l "/dev/video${VIDEO_NR}" 2>/dev/null || true
}

persist_v4l2loopback() {
  log "Persisting v4l2loopback across reboot..."
  echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null

  cat <<EOF | sudo tee /etc/modprobe.d/studiocast-v4l2loopback.conf >/dev/null
# StudioCast v4l2loopback options
options v4l2loopback video_nr=${VIDEO_NR} card_label="${LABEL}" exclusive_caps=${EXCLUSIVE_CAPS}
EOF

  log "Wrote:"
  log "  /etc/modules-load.d/v4l2loopback.conf"
  log "  /etc/modprobe.d/studiocast-v4l2loopback.conf"
  log "You can verify after reboot with:"
  log "  modinfo v4l2loopback | head"
  log "  ls -l /dev/video${VIDEO_NR}"
}

while [[ $# -gt 0 ]]; do
  if [[ "$PARSE_MAXINE_ARGS" -eq 1 ]]; then
    MAXINE_ARGS+=("$1"); shift; continue
  fi

  case "$1" in
    --deps) DO_DEPS=1; shift ;;
    --v4l2loopback) DO_V4L2=1; DO_DEPS=1; shift ;;
    --load-loopback) DO_LOAD_LOOP=1; shift ;;
    --persist-loopback) DO_PERSIST_LOOP=1; shift ;;
    --video-nr) VIDEO_NR="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --exclusive-caps) EXCLUSIVE_CAPS="${2:-}"; shift 2 ;;
    --onnxruntime-version) ORT_VERSION="${2:-}"; shift 2 ;;
    --onnxruntime-flavor) ORT_FLAVOR="${2:-}"; shift 2 ;;
    --onnxruntime-arch) ORT_ARCH="${2:-}"; shift 2 ;;
    --vulkan-runtime) DO_VULKAN_RUNTIME=1; shift ;;
    --mesa-vulkan) DO_MESA_VULKAN=1; DO_VULKAN_RUNTIME=1; shift ;;
    --shader-tools) DO_SHADER_TOOLS=1; shift ;;
    --build) DO_BUILD=1; shift ;;
    --build-dir) BUILD_DIR="${2:-}"; shift 2 ;;
    --build-type) BUILD_TYPE="${2:-}"; shift 2 ;;
    --maxine) DO_MAXINE=1; shift ;;
    -y|--yes) YES=1; shift ;;
    --) PARSE_MAXINE_ARGS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ "${ORT_FLAVOR}" != "cpu" && "${ORT_FLAVOR}" != "gpu" ]]; then
  echo "[setup] ERROR: --onnxruntime-flavor must be one of: cpu|gpu (got: '${ORT_FLAVOR}')" >&2
  exit 2
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    log "Warning: tuned for Ubuntu. Detected ID=${ID:-unknown}."
  else
    log "Detected Ubuntu ${VERSION_ID:-unknown} (${VERSION_CODENAME:-unknown})."
  fi
fi

if [[ "$YES" -eq 1 ]]; then
  APT_ARGS+=("-y")
  # apt-get flags for noninteractive scripting
  APT_GET_ARGS+=("-y")
fi

# IMPORTANT: heal dpkg state before any apt installs.
fix_broken_dkms_if_newer_kernel_module_exists

if [[ "$DO_DEPS" -eq 1 ]]; then
  log "Installing build/runtime dependencies..."
  apt_install \
    build-essential cmake ninja-build pkg-config \
    git curl ca-certificates tar \
    qt6-base-dev qt6-base-dev-tools qt6-tools-dev qt6-tools-dev-tools \
    qtbase5-dev \
    libxkbcommon-dev \
    libpulse-dev libpulse0 pulseaudio-utils \
    clang clang-format clang-tidy \
    v4l-utils \
    libblas-dev liblapack-dev \
    libdlib-dev libsqlite3-dev \
    libjpeg-turbo8 libjpeg-turbo8-dev \
    libpng-dev

  ensure_onnxruntime_available
fi

if [[ "$DO_VULKAN_RUNTIME" -eq 1 ]]; then
  install_optional_vulkan_runtime
fi

if [[ "$DO_SHADER_TOOLS" -eq 1 ]]; then
  install_optional_shader_tools
fi

if [[ "$DO_V4L2" -eq 1 ]]; then
  ensure_v4l2loopback_available
fi

if [[ "$DO_LOAD_LOOP" -eq 1 ]]; then
  if ! have_module; then
    ensure_v4l2loopback_available
  fi
  load_v4l2loopback_now
fi

if [[ "$DO_PERSIST_LOOP" -eq 1 ]]; then
  if ! have_module; then
    ensure_v4l2loopback_available
  fi
  persist_v4l2loopback
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
  log "Configuring + building into: $BUILD_DIR (type: $BUILD_TYPE)"
  cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON \
    -DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON
  cmake --build "$BUILD_DIR"
  log "Built. Useful commands:"
  echo "  $BUILD_DIR/studiocast --version"
  echo "  $BUILD_DIR/studiocastd"
  echo "  $BUILD_DIR/studiocastctl status"
  echo "  $BUILD_DIR/studiocast-maxine install-hints"
fi

if [[ "$DO_MAXINE" -eq 1 ]]; then
  log "Running Maxine setup helper..."
  ./scripts/setup/maxine.sh --build-dir "$BUILD_DIR" "${MAXINE_ARGS[@]}"
fi

log "Done."
