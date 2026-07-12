# StudioCast Developer Guide

This guide collects the source-build, architecture, daemon, IPC, model, and
testing notes that are too detailed for the top-level README.

For user-facing setup and usage, start with [../README.md](../README.md).

## Build from source

StudioCast currently targets Ubuntu 22.04 and 24.04. The setup helper supports
Ubuntu-family distributions and installs the common build/runtime dependencies,
ONNX Runtime, and v4l2loopback support.

One-shot development setup:

```bash
./scripts/setup.sh --deps --v4l2loopback --load-loopback --persist-loopback
./scripts/setup.sh --build --build-dir ./build --build-type Debug
```

Manual CMake build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target studiocast studiocastd studiocastctl
```

Useful development commands:

```bash
./build/studiocast --version
./build/studiocast-installer --version
./build/studiocastd
./build/studiocastctl status --pretty
ctest --test-dir build --output-on-failure
./scripts/dev/format.sh
```

CLion often uses `cmake-build-debug/` or `cmake-build-release/` instead of
`build/`. Substitute the build directory in commands as needed.

See [SETUP.md](SETUP.md) for the longer install notes and v4l2loopback fallback
details.

## Versioning

The canonical project version lives in the top-level
[../VERSION](../VERSION) file as `MAJOR.MINOR.PATCH`. CMake reads that file
before `project()`, so `PROJECT_VERSION`, the generated
`studiocast/version.h`, CLI `--version` output, daemon status, and GUI About
surfaces all use the same value.

The current automatic versioning line starts at `0.2.0`. On every normal push
to `master`, `.github/workflows/version-bump.yml` increments the patch component
and commits the updated `VERSION` file back to `master` with a `[skip ci]`
message. The workflow skips bot commits and skips pushes that already changed
`VERSION`, which prevents commit loops and lets maintainers intentionally set a
new baseline such as `0.3.0`.

If `master` is protected, repository settings must allow the GitHub Actions bot
to push the version-bump commit.

For releases, wait for the automatic version-bump commit to land on `master`,
then tag that commit with the matching version:

```bash
git fetch origin master --tags
git checkout origin/master
git tag -a v0.2.1 -m "StudioCast 0.2.1"
git push origin v0.2.1
```

If maintainers need a new minor or major line, update `VERSION` in a normal
change, merge it to `master`, and tag the resulting commit if it is a release.

## Repo layout

- [../CMakeLists.txt](../CMakeLists.txt): build graph, options, executable
  targets, and test targets.
- [../src/core](../src/core): shared non-Qt core code for config, IPC, audio,
  video, effects, CUDA, Maxine, ONNX Runtime, and utility code.
- [../src/daemon](../src/daemon): `studiocastd`, the background service that
  owns runtime state and device orchestration.
- [../src/gui](../src/gui): Qt GUI controller.
- [../src/tools](../src/tools): command-line helpers.
- [../installer/gui](../installer/gui): standalone Qt installer wizard.
- [../installer/backend](../installer/backend): scriptable installer backend
  used by the GUI and CLI fallback.
- [../tests](../tests): unit and integration-style tests that do not require
  full desktop hardware workflows.
- [../scripts](../scripts): setup, install, uninstall, model, and developer
  helper scripts. See [../scripts/README.md](../scripts/README.md).
- [../resources/model_packs](../resources/model_packs): metadata templates for
  curated model packs. Model binaries are downloaded separately.
- [../packaging/systemd/user](../packaging/systemd/user): systemd user service
  template for `studiocastd`.
- [../packaging/appimage](../packaging/appimage): release packaging scaffold
  for the standalone GUI installer bundle.
- [../docs](../docs): architecture, setup, model installation, manual testing,
  trademark, roadmap, and design notes.

## Main binaries and tools

| Binary | Purpose |
| --- | --- |
| `studiocast` | Qt GUI controller for users. |
| `studiocast-installer` | Qt installer wizard that calls the scriptable backend. |
| `studiocastd` | Background daemon that owns camera/audio services, config, status, and effect availability. |
| `studiocastctl` | CLI client for daemon status, config, effects, audio/video controls, and debug reports. |
| `studiocast-open` | Open Video/Open Audio model path, listing, validation, and benchmark helper. |
| `studiocast-maxine` | Maxine path, GPU, doctor, install-hints, and smoke-test helper. |
| `studiocast-probe` | System probe/diagnostic helper. |
| `studiocast-audio` | Audio diagnostics/helper tool. |
| `studiocast-video` | Video diagnostics/helper tool. |

The install helper can symlink built binaries into `~/.local/bin` for a user
service workflow:

```bash
./scripts/install.sh user-service --build-dir ./build --yes
```

## Installer architecture

The installer is intentionally split into a GUI front end and a scriptable
backend:

- `studiocast-installer` is a separate Qt Widgets target. It does not require
  StudioCast to already be installed and refuses to run as root.
- `installer/backend/studiocast-installer-backend` owns OS detection, planning,
  install/update/repair/uninstall/clean-install execution, and manifest writes.
- `scripts/installer.sh` is the stable CLI entrypoint for CI, SSH, recovery,
  and debugging.
- Existing `scripts/setup.sh`, `scripts/install.sh`, and `scripts/uninstall.sh`
  remain compatibility entrypoints. The backend delegates dependency setup,
  v4l2loopback setup, binary linking, model installs, and systemd user service
  setup to those scripts instead of duplicating their logic.

Backend examples:

```bash
./scripts/installer.sh detect-os --json
./scripts/installer.sh status --json
./scripts/installer.sh plan install --json
./scripts/installer.sh install --yes
./scripts/installer.sh update --source-dir /path/to/release-source --yes
./scripts/installer.sh update --release-archive ~/Downloads/studiocast.tar.gz --yes
./scripts/installer.sh repair --yes
./scripts/installer.sh uninstall --yes
./scripts/installer.sh clean-install --yes
```

The GUI installer exposes Open Vulkan as four independent, opt-in choices:
build with the runtime-loaded backend, install loader/diagnostic packages,
install Mesa Intel/AMD ICD packages, and install developer shader tools. The
corresponding backend flags are `--open-vulkan`, `--vulkan-runtime`,
`--mesa-vulkan`, and `--shader-tools`. Install, update, repair, and clean-install
plans must preserve those selections in both the review text and the executed
backend command. Vulkan package selection remains useful in repair flows even
when the full dependency bundle is skipped.

Installer review text must state that Open Vulkan is optional and runtime-loaded
and still requires a working GPU driver/ICD. It must not equate loader/device
availability with production inference parity: production Vulkan
virtual-background matting remains blocked until a device-resident Vulkan
inference runtime is implemented.

The installer backend supports Ubuntu 22.04/Jammy, Ubuntu 24.04/Noble, and Linux
Mint when `/etc/os-release` exposes a reliable Ubuntu base. Mint
`UBUNTU_CODENAME=jammy` maps to Ubuntu 22.04; `UBUNTU_CODENAME=noble` maps to
Ubuntu 24.04. If that field is absent, Mint 21.x maps to Jammy and Mint 22.x
maps to Noble.

The install manifest lives at:

```text
~/.local/share/studiocast/install-manifest.json
```

It records installed version, source/build/install paths, installed binaries,
systemd user service path/status, dependency install method, install timestamp,
and whether config/models/logs/cache were preserved. Update and repair prefer
manifest paths unless a source directory or release archive is explicitly
selected. Uninstall removes app symlinks, service files, desktop entries, and
runtime artifacts. Clean install preserves user config, downloaded models, logs,
and cache unless `--remove-user-data` is explicitly selected.

Release packaging:

- `packaging/appimage/build_appimage.sh` configures an isolated Release build,
  builds only `studiocast-installer`, installs the `Installer` CMake component
  into an AppDir, adds desktop/icon metadata, archives the AppDir, and writes
  SHA256 checksums.
- The AppDir layout places the backend at
  `usr/share/studiocast/installer/studiocast-installer-backend`, which is the
  installed path the GUI already probes relative to the installer binary.
- The same packaging script creates `StudioCast-<version>-source.tar.gz` from
  `HEAD` with `git archive` when available, stages it at
  `usr/share/studiocast/source/StudioCast-<version>-source.tar.gz`, and leaves
  the standalone source archive in `dist/appimage/`.
- Release AppImages are self-contained for the installer GUI, backend, and
  matching source archive. Runtime dependencies still come from supported system
  packages, ONNX Runtime/model helpers, optional SDK assets, and the installer
  backend scripts.
- Local packaging does not download tools. If `linuxdeploy` and
  `linuxdeploy-plugin-qt` are available, the script also creates
  `StudioCast-Installer-<version>-<arch>.AppImage`; otherwise it leaves the
  staged AppDir tarball as the local artifact.
- Release CI is in `.github/workflows/release-packaging.yml`. It runs only from
  `workflow_dispatch` or a published GitHub Release event, downloads AppImage
  packaging tools from `packaging/appimage/tools.lock`, verifies each tool's
  SHA256 before making it executable, requires AppImage generation, and uploads
  the installer bundle, AppDir archive, source archive, and checksum file as
  workflow artifacts. It does not tag commits or publish release assets by
  itself.
- The packaged installer is still a source-build installer. The GUI defaults to
  the bundled source archive, and users can still point it at another release
  archive or a local checkout.

First release checklist:

1. Merge the release change to `master`.
2. Wait for `.github/workflows/version-bump.yml` to commit the next `VERSION`
   value back to `master`.
3. Fetch the updated branch and tags:

   ```bash
   git fetch origin master --tags
   git checkout origin/master
   ```

4. Confirm the release version:

   ```bash
   cat VERSION
   ```

5. Create and push an annotated tag that matches `VERSION`:

   ```bash
   version="$(cat VERSION)"
   git tag -a "v${version}" -m "StudioCast ${version}"
   git push origin "v${version}"
   ```

6. In GitHub, create and publish a Release for that tag. Publishing the Release
   triggers `.github/workflows/release-packaging.yml`.
7. After release packaging finishes, download the
   `studiocast-gui-installer-ubuntu-22.04` workflow artifact and attach the
   AppImage, AppDir archive, source archive, and SHA256 file to the GitHub
   Release.

Pinned AppImage tool updates:

1. Choose fixed release asset URLs for `linuxdeploy` and
   `linuxdeploy-plugin-qt`; do not pin to upstream `continuous` URLs.
2. Download each AppImage and record `sha256sum <file>`.
3. Update `packaging/appimage/tools.lock` with the matching version, URL, and
   SHA256 values in one change.
4. Run release packaging or a workflow-dispatch dry run so CI verifies the
   checksums before either packaging tool is executed.

Maintainer command:

```bash
packaging/appimage/build_appimage.sh --clean
```

For release-equivalent local validation with preinstalled packaging tools:

```bash
packaging/appimage/build_appimage.sh --clean --appimage-required
```

## Daemon architecture

`studiocastd` is the runtime authority. The GUI and CLI are controllers that
communicate with the daemon rather than directly owning the production device
state.

The daemon is responsible for:

- Loading persisted config from the XDG config location.
- Publishing status for the GUI and CLI.
- Keeping the virtual camera device available through v4l2loopback.
- Starting and stopping video processing based on consumer detection.
- Managing virtual microphone and speaker state.
- Resolving safe audio sources and targets.
- Computing effect and engine availability.
- Persisting canonical audio and video effect JSON.

Heavy video processing should be consumer-gated: the virtual camera can remain
available, but the camera pipeline should not keep doing expensive work when no
external app or GUI preview is consuming it. Audio routing distinguishes speaker
pass-through loopback from processed speaker effects in daemon status.

Existing architecture notes live in [ARCHITECTURE.md](ARCHITECTURE.md).

## IPC and socket behavior

The control plane is a small line-based Unix domain socket protocol.

Default socket path:

```text
$XDG_RUNTIME_DIR/studiocast/studiocastd.sock
```

If `XDG_RUNTIME_DIR` is unavailable, the socket helper falls back to a
per-user temporary runtime directory. The authoritative path is reported in
`studiocastctl status`.

Relevant source files:

- [../src/core/ipc/daemon_socket.cpp](../src/core/ipc/daemon_socket.cpp)
- [../src/core/ipc/daemon_client.cpp](../src/core/ipc/daemon_client.cpp)
- [../src/core/ipc/daemon_server.cpp](../src/core/ipc/daemon_server.cpp)
- [../src/daemon/studiocastd_main.cpp](../src/daemon/studiocastd_main.cpp)

Common daemon commands:

- `GET_STATUS`: full runtime status and diagnostics.
- `GET_CONFIG`: canonical video effects JSON.
- `GET_AUDIO_CONFIG`: canonical audio config/effects JSON.
- `SET_ENABLED`: enable/disable video pipeline intent.
- `SET_VIDEO_CONFIG`: patch video device and pipeline options.
- `SET_AUDIO_CONFIG`: patch audio device, routing, and effect options.
- `SET_VIDEO_EFFECTS_JSON`: patch canonical video effects JSON.

Use `studiocastctl` when possible instead of manually writing socket messages:

```bash
./build/studiocastctl status --pretty
./build/studiocastctl effects get
./build/studiocastctl effects set --file effects.json
./build/studiocastctl audio get
./build/studiocastctl audio set --file audio.json
```

## Audio pipeline overview

Audio code lives mainly under [../src/core/audio](../src/core/audio).

Important pieces:

- `virtual_mic.*`: StudioCast virtual microphone management.
- `virtual_speaker.*`: StudioCast virtual speaker management.
- `virtual_audio_service.*`: high-level virtual audio service coordination.
- `audio_pipeline.*`: real-time processing path when libpulse-simple is
  available.
- `audio_device_safety.*`: source/target validation to avoid feedback loops.
- `audio_backend_resolver.*`: backend selection for audio effects.
- `effects/broadcast_audio_effects*`: canonical audio effect model and JSON.
- `open_audio/*`: ONNX Runtime model discovery, diagnostics, and processing.
- `maxine/afx/*`: optional Maxine Audio Effects integration.

Safety rule: production audio config should go through daemon IPC. The daemon
and audio pipeline reject StudioCast virtual microphone sources, Pulse monitor
sources, and StudioCast virtual speaker targets where they could create feedback
or self-capture. `source: "auto"` resolves to a safe physical microphone when
possible; otherwise status should report an actionable error.

Canonical audio effects are persisted under `audio.effects.json` in the daemon
config.

## Video, effects, and model backends

Video code lives mainly under [../src/core/video](../src/core/video), with
backend-specific code under [../src/core/open_video](../src/core/open_video),
[../src/core/maxine](../src/core/maxine), and [../src/core/cuda](../src/core/cuda).

Key concepts:

- v4l2loopback provides the writable virtual camera device.
- `v4l2_capture.*` reads physical camera frames.
- `v4l2_writer.*` writes output frames to the virtual camera.
- `virtual_camera_service.*` coordinates the virtual camera lifecycle.
- `camera_pipeline.*` handles capture, effects, scaling, and output.
- `effects/broadcast_effects.*` defines the canonical video effect state.
- `effects/broadcast_effect_contract.h` defines stable effect IDs, parameter
  IDs, and ranges for IPC and JSON.

Effect engine preference values:

- `auto` / `auto_select`: prefer Maxine when available, otherwise use Open CUDA
  for supported effects.
- `maxine`: force the Maxine backend where supported.
- `open_cuda`: force the Open Video/Open CUDA backend where supported.

Video compute preference values:

- `auto`: prefer usable CUDA on NVIDIA systems, otherwise resolve to the next
  supported backend or CPU/pass-through.
- `cpu`: force CPU/pass-through behavior for GPU-backed effects.
- `cuda`: request Open CUDA; missing CUDA/ORT/model support must be reported as
  unavailable or degraded.
- `vulkan`: request Open Vulkan. This must not silently run CUDA; if Vulkan is
  unavailable, status reports a Vulkan fallback/degraded reason and the active
  backend reflects CPU/pass-through where applicable.

Availability is daemon-owned. The GUI should use `GET_STATUS` instead of trying
to infer local engine or model state. Daemon status reports nested engine
diagnostics such as `engines.maxine`, `engines.open_cuda`, and
`engines.open_vulkan` and `engines.open_audio`, plus compatibility aliases
where present.

The normalized video backend status lives under `video.compute` and includes:

- `preference`, `resolved_backend`, `active_backend`
- `fallback_reason`, `degraded_reason`, and `fallback`
- `active_engines`, `unavailable_reasons`, and `provider`
- compact `cpu_tails` and `transfers` rollups

Keep this status built from cached setup-time diagnostics and pipeline counters.
Do not add provider/package/runtime checks, JSON construction, or probe logic to
the frame loop.

Open CUDA video effects are GPU-only. If CUDA, ONNX Runtime, or model packs are
missing, the effects should be marked unavailable rather than silently falling
back to CPU.

Open Vulkan is optional and runtime-loaded. Build with
`-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON` to compile it. Ubuntu-family setup can
install loader/diagnostic packages with `./scripts/setup.sh --vulkan-runtime`
and Mesa Intel/AMD ICD packages with `--mesa-vulkan`; these packages do not
promise hardware support. Runtime/device availability also does not imply
production Vulkan virtual-background matting readiness: until a
device-resident Vulkan matting runtime is available, daemon status must report
Open Vulkan virtual background as blocked/degraded instead of silently running
CUDA.

The daemon persists Vulkan adapter intent as `video.vulkan.device` and
`video.vulkan.allow_cpu` in `daemon.conf`. `auto` uses hardware-first selection
and retains `STUDIOCAST_VULKAN_DEVICE_INDEX` as a compatibility override. A
saved `v1:...` identity (vendor ID, device ID, device type, and normalized
device name) takes precedence over the run-local index. Missing saved devices
and indistinguishable duplicate identities fail closed; they are never replaced
silently. CPU Vulkan remains an explicit software fallback. Adapter changes
apply to subsequently initialized Vulkan devices, so restart the daemon before
relying on a changed adapter for a running pipeline.

Canonical video effects are persisted under `video.effects.json` in the daemon
config.

## Model installation and validation

StudioCast discovers runtime model packs from user-local XDG data directories.
Source-tree pack entries live under
[../resources/model_packs](../resources/model_packs) and may include metadata,
license summaries, placeholders, and intentionally curated artifacts. Install
scripts fetch curated runtime assets into XDG data directories.

Policy:

- [MODEL_POLICY.md](MODEL_POLICY.md)

Open Video / Open CUDA:

```bash
./scripts/install.sh open-video-models --list
./scripts/install.sh open-video-models
./build/studiocast-open video-list-models
./build/studiocast-open video-self-test --model-id <id>
```

Details:

- [open_source_video_models_install.md](open_source_video_models_install.md)
- [open_source_video_model_conversion_to_onnx.md](open_source_video_model_conversion_to_onnx.md)
- [open_video_shared_inference.md](open_video_shared_inference.md)
- [how_to_add_your_own_model.md](how_to_add_your_own_model.md)

Open Audio:

```bash
./scripts/install.sh open-audio-models --list
./scripts/install.sh open-audio-models
./build/studiocast-open audio-list-models
./build/studiocast-open audio-self-test --model-id fastenhancer_s_vd_v1
./build/studiocast-open audio-bench --effect noise_removal --model-id fastenhancer_s_vd_v1 --seconds 10
```

Details:

- [open_source_audio_models_install.md](open_source_audio_models_install.md)

NVIDIA Maxine:

```bash
./build/studiocast-maxine init
./build/studiocast-maxine install-hints
./build/studiocast-maxine doctor
```

Details:

- [maxine_install.md](maxine_install.md)

StudioCast does not redistribute NVIDIA SDK assets, feature packs, model files,
or keys. Users must obtain those from NVIDIA and comply with NVIDIA's license
terms.

## Manual testing and debugging

Hardware, GUI, v4l2loopback, desktop app, GPU, and audio-routing workflows need
manual validation. Use [MANUAL_TESTING.md](MANUAL_TESTING.md) for regression
passes.

High-value commands:

```bash
./build/studiocastctl status --pretty
./build/studiocastctl debug-report --out studiocast-debug-report.txt
v4l2-ctl --list-devices
v4l2-ctl --all -d /dev/video10
pactl info
pactl list short sources
pactl list short sinks
pactl list short modules
journalctl --user -u studiocastd.service -f
```

Useful checks:

- Start `studiocastctl status` before the daemon and confirm it fails quickly
  with a clear socket error.
- Start `studiocastd`, then confirm status reports the configured virtual
  camera.
- Open OBS or a browser/WebRTC test page and confirm consumer detection starts
  the video pipeline.
- Close all consumers and confirm heavy processing returns to idle.
- Generate a debug report after failures.

## Benchmarks and hardware CI

Benchmark tools are opt-in:

```bash
cmake -S . -B build-bench -G Ninja \
  -DSTUDIOCAST_BUILD_BENCHMARKS=ON \
  -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON
cmake --build build-bench --target \
  studiocast-rgb-yuyv-bench \
  studiocast-cuda-transfer-bench \
  studiocast-resize-backend-bench
```

Coverage:

- `studiocast-rgb-yuyv-bench`: CPU color conversion, CPU resize, CPU
  background-remove primitives.
- `studiocast-cuda-transfer-bench`: CUDA upload, download, and roundtrip
  transfer rows.
- `studiocast-resize-backend-bench`: CPU/CUDA/Vulkan resize comparisons when
  each backend is compiled and available.

CSV output includes warmup/iteration counts, avg/p95/p99/min/max timing, status
(`ok` or `skipped`), and skip reasons for unavailable GPU backends. Hardware
benchmarks must stay out of default push CI. The CI workflow keeps ordinary jobs
no-GPU-safe and exposes workflow-dispatch smoke jobs for CUDA/NVIDIA and
Vulkan/NVIDIA, Vulkan/AMD, and Vulkan/Intel runners.

## Packaging and systemd notes

The systemd user service template is:

- [../packaging/systemd/user/studiocastd.service](../packaging/systemd/user/studiocastd.service)

The install helper:

```bash
./scripts/install.sh user-service --build-dir ./build --yes
```

What it does:

- Creates or refreshes `~/.local/bin` symlinks to built StudioCast binaries.
- Copies the service file to `~/.config/systemd/user/studiocastd.service`.
- Runs `systemctl --user daemon-reload`.
- Enables and starts `studiocastd.service`.

Service commands:

```bash
systemctl --user status studiocastd.service
systemctl --user restart studiocastd.service
journalctl --user -u studiocastd.service -f
```

The current packaging flow is suitable for development and MVP testing. Treat
distribution packaging and polished non-developer install flows as future work.

## Deeper docs

- [SETUP.md](SETUP.md): source setup, dependencies, v4l2loopback, and optional
  backend setup.
- [ARCHITECTURE.md](ARCHITECTURE.md): canonical effect model and daemon-owned
  availability notes.
- [MANUAL_TESTING.md](MANUAL_TESTING.md): hardware and GUI manual regression
  plan.
- [ROADMAP.md](ROADMAP.md): project direction and planned work.
- [TRADEMARKS.md](TRADEMARKS.md): affiliation and trademark note.
- [../CONTRIBUTING.md](../CONTRIBUTING.md): contribution conventions.
- [../SECURITY.md](../SECURITY.md): security reporting.
