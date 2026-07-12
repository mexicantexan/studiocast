# Setup / Install (Ubuntu 22.04+)

This repo supports both a manual source-build flow and a StudioCast installer
wizard target. The GUI installer is the polished user path for releases; the
scripts below remain the CLI fallback for CI, SSH, recovery, and debugging.

## GUI installer wizard

Release downloads may include:

- `StudioCast-Installer-<version>-<arch>.AppImage`: GUI installer bundle when
  AppImage packaging tools succeeded in CI.
- `StudioCast-Installer-<version>-<arch>.AppDir.tar.gz`: staged AppDir archive
  for inspection or fallback testing.
- `StudioCast-<version>-source.tar.gz`: the same source archive bundled in the
  installer and also uploaded separately for inspection or manual use.
- `StudioCast-Installer-<version>-<arch>.sha256`: checksums for the release
  artifacts.

Run the GUI as your normal user, not with `sudo`. If using the AppImage, make it
executable and launch it:

```bash
chmod +x StudioCast-Installer-<version>-<arch>.AppImage
./StudioCast-Installer-<version>-<arch>.AppImage
```

The packaged installer includes its backend at
`usr/share/studiocast/installer/studiocast-installer-backend`, which matches the
GUI lookup path. It also includes the matching release source archive at
`usr/share/studiocast/source/StudioCast-<version>-source.tar.gz`; the GUI
automatically selects "Build from selected release archive" with that archive
prefilled. You can still choose a different release archive manually or build
from a local checkout.

The release AppImage/AppDir is self-contained for the installer GUI, backend,
and matching source archive. Runtime dependencies such as Qt build packages,
v4l2loopback support, PulseAudio tools, ONNX Runtime, and optional model/SDK
assets are installed or checked through supported system packages and the
backend scripts.

Build and run the installer from a checkout:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target studiocast-installer
./build/studiocast-installer
```

The GUI calls the scriptable
backend at `installer/backend/studiocast-installer-backend`, which uses the
existing setup/install/uninstall helpers for privileged package/module steps,
builds, binary links, and the systemd user service.

On the **Service and Optional Components** page, users can separately select:

- **Build StudioCast with Open Vulkan support** (`--open-vulkan`).
- **Install Vulkan loader and diagnostic packages** (`--vulkan-runtime`).
- **Install Mesa Intel/AMD Vulkan ICDs** (`--mesa-vulkan`).
- **Install shader developer tools** (`--shader-tools`), which are only needed
  to validate or regenerate the committed SPIR-V shaders.

These choices are available for install, update, repair, and clean-install
workflows. They are opt-in. Open Vulkan is runtime-loaded, and package
installation alone does not make a Vulkan GPU usable: the system still needs a
working GPU driver/ICD exposing a compute-capable Vulkan device. The review page
shows the selected backend flags and package operations before anything runs.

CLI equivalents:

```bash
./scripts/installer.sh detect-os
./scripts/installer.sh status
./scripts/installer.sh plan install
./scripts/installer.sh install --yes
./scripts/installer.sh update --source-dir /path/to/studiocast-release --yes
./scripts/installer.sh repair --yes
./scripts/installer.sh uninstall --yes
./scripts/installer.sh clean-install --yes
```

Supported installer OS bases:

- Ubuntu 22.04 / Jammy
- Ubuntu 24.04 / Noble
- Linux Mint when `/etc/os-release` exposes a reliable Ubuntu base:
  `UBUNTU_CODENAME=jammy` maps to Ubuntu 22.04 and
  `UBUNTU_CODENAME=noble` maps to Ubuntu 24.04. Mint 21.x and 22.x are mapped
  to those bases if `UBUNTU_CODENAME` is missing.
- Other Ubuntu-family distributions may also pass installer checks when
  `/etc/os-release` declares `ID_LIKE=ubuntu` and exposes
  `UBUNTU_CODENAME=jammy` or `UBUNTU_CODENAME=noble`.

Unsupported distros fail clearly with the detected `/etc/os-release` fields and
should use the manual source-build flow below.

The installer writes:

```text
~/.local/share/studiocast/install-manifest.json
```

The manifest records installed version, source/build/install paths, installed
binaries, user service path/status, dependency setup method, install timestamp,
and whether user config/model/log/cache data was preserved. Update, repair,
uninstall, and clean-install workflows use this manifest when available.

Clean install removes app files and the user service before reinstalling. It
preserves user config, downloaded model packs, logs, and cache by default; the
GUI and backend require an explicit `--remove-user-data` choice before deleting
those XDG directories.

## 1) Install dependencies + v4l2loopback

```bash
./scripts/setup.sh --deps --v4l2loopback --load-loopback --persist-loopback
```

This installs build dependencies (Qt6/CMake/Ninja/etc), ONNX Runtime, and runtime dependencies (v4l2loopback tools, v4l-utils),
then creates a virtual camera device (by default at `/dev/video10` with label **StudioCast Camera**).
The setup helper prefers a kernel-provided/prebuilt v4l2loopback module and only falls back to `v4l2loopback-dkms` when the running kernel does not already provide one.
The ONNX Runtime flavor is `gpu` only when `nvidia-smi` works or when you pass `--onnxruntime-flavor gpu`; otherwise the helper installs the CPU flavor.

MJPEG decode uses **libjpeg-turbo** (via CMake `FindJPEG`). If you are installing dependencies manually:

```bash
sudo apt install libjpeg-turbo8 libjpeg-turbo8-dev
```

Verify:

```bash
v4l2-ctl --list-devices
```

### v4l2loopback module install fallback

Ubuntu kernels may already ship a signed/prebuilt `v4l2loopback.ko`. StudioCast
checks that first and avoids DKMS when it is available. If the helper falls back
to `v4l2loopback-dkms` and that DKMS build fails against a newer kernel, install
v4l2loopback from upstream source, then rerun the StudioCast setup with
`--load-loopback --persist-loopback`:

```bash
git clone https://github.com/v4l2loopback/v4l2loopback.git /tmp/v4l2loopback
cd /tmp/v4l2loopback
VERSION=$(grep 'PACKAGE_VERSION' dkms.conf | cut -d'"' -f2)
sudo mkdir -p /usr/src/v4l2loopback-${VERSION}
sudo cp -r ./* /usr/src/v4l2loopback-${VERSION}/
sudo dkms add v4l2loopback/${VERSION}
sudo dkms autoinstall
```

If you have the Ubuntu DKMS package installed and want to keep apt from
replacing the upstream source copy, hold it:

```bash
sudo apt-mark hold v4l2loopback-dkms
```

## 2) Build StudioCast

```bash
./scripts/setup.sh --build --build-type Debug
```

Or manually:

```bash
cmake -S . -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
```

## 3) Optional: Open CUDA backend (no Maxine required)

The Open CUDA backend (`open_cuda`) runs GPU-only virtual background effects via ONNX Runtime (CUDA EP), using
user-supplied model packs.

Install/usage docs:

- `docs/open_cuda_install.md`

On Ubuntu 22.04+, the helper script that installs ONNX Runtime is:

```bash
./scripts/setup.sh --deps
```

`--deps` ensures ONNX Runtime is available via `pkg-config onnxruntime` by installing the upstream tarball under
`/opt/studiocast/onnxruntime/<version>/...` and configuring runtime linking via `ldconfig`.
For Open CUDA, use a working NVIDIA driver and install the GPU ONNX Runtime flavor explicitly when auto-detection cannot see `nvidia-smi`:

```bash
./scripts/setup.sh --deps --onnxruntime-flavor gpu
```

## 3a) Optional: Open Vulkan backend

The Open Vulkan backend is optional and disabled at build time unless requested:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON
```

Vulkan runtime packages are also optional. On Ubuntu-family systems:

```bash
./scripts/setup.sh --vulkan-runtime
```

For Intel/AMD Mesa ICDs, add:

```bash
./scripts/setup.sh --vulkan-runtime --mesa-vulkan
```

These packages provide the Vulkan loader and diagnostics (`libvulkan1`, `vulkan-tools`, and optionally `mesa-vulkan-drivers`). They do not guarantee hardware support. You still need a GPU driver/ICD that exposes a compute-capable Vulkan device. Developer shader tools are separate:

```bash
./scripts/setup.sh --shader-tools
```

Explicit Vulkan selection does not silently run CUDA. If Vulkan is requested but the build/runtime/device path is unavailable, daemon status reports the Vulkan fallback/degraded reason and the active backend is CPU/pass-through where applicable.

Open Vulkan install/build visibility is ahead of full effect parity. In
particular, production Vulkan virtual-background matting remains blocked until
a device-resident Vulkan inference runtime replaces the current stub. Installing
the Vulkan loader or Mesa ICDs must not be read as a claim of CUDA/NVIDIA effect
parity.

## 3b) Optional: Open Audio backend (no Maxine required)

The Open Audio backend (`open_audio`) runs **microphone** effects (noise removal + “studio voice”-style enhancement)
via ONNX Runtime, using user-supplied model packs.

Install/usage docs:

- `docs/open_source_audio_models_install.md`

Install the curated model pack:

```bash
./scripts/install.sh open-audio-models
```

Verify:

```bash
./cmake-build-debug/studiocast-open audio-list-models
./cmake-build-debug/studiocast-open audio-self-test --model-id fastenhancer_m_vd_v1
```

## 4) Optional: Install NVIDIA Maxine SDK + features

StudioCast does **not** ship or redistribute NVIDIA Maxine SDK assets. You must obtain them yourself
from NVIDIA and comply with NVIDIA's license terms.

Use the helper tool to print authoritative paths and install commands:

```bash
./cmake-build-debug/studiocast-maxine init
./cmake-build-debug/studiocast-maxine install-hints
```

Current Linux Maxine SDK builds typically expose `libVideoFX.so` for VFX and
`libnvARPose.so` for AR. StudioCast now auto-detects those names directly, but
you still need to run the SDK-provided `install_feature.sh` steps so the effect
models and feature libraries are installed.

### Optional: automate extraction + feature install

If you already downloaded the SDK tarballs, you can extract them into the expected layout and install
features (models/libs) via NGC:

```bash
export NGC_CLI_API_KEY="..."   # do not commit this
export NGC_API_KEY="..."       # do not commit this
./scripts/setup/maxine.sh \
  --vfx-tar ~/Downloads/NVIDIA_VFX_SDK_linux_*.tar.gz \
  --ar-tar  ~/Downloads/NVIDIA_AR_SDK_linux_*.tar.gz \
  --afx-tar ~/Downloads/Audio_Effects_SDK.tar.gz \
  --install-features --install-afx-features --build-dir ./cmake-build-debug
```

By default, `--install-afx-features` downloads the MVP AFX feature set (AEC + Superres). To customize:

```bash
./scripts/setup/maxine.sh --install-afx-features --afx-effects "superres-16k_to_48k,superres-8k_to_16k,aec-16k,aec-48k"
```

This runs the SDK-provided `install_feature.sh` scripts under the hood.

For AFX features, the helper uses the SDK-provided `download_features.sh` script and requires `NGC_API_KEY`.

## 5) Run daemon + use in OBS

```bash
./cmake-build-debug/studiocastd
./cmake-build-debug/studiocastctl status
```

In OBS: select **StudioCast Camera** as a camera source.

## Support bundle

If something doesn't work:

```bash
./cmake-build-debug/studiocastctl debug-report --out studiocast-debug-report.txt
```

Attach that file in GitHub issues.
