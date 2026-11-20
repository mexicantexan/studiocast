# StudioCast

StudioCast is an open-source Linux application that provides a virtual camera,
virtual microphone, and virtual speakers with real-time effects for video calls,
streaming, and recording.

## It's Pretty Awesome NGL

Most Linux broadcast-effect projects prove one cool trick or half bake the broader idea. StudioCast is on another level and constantly improving!

- Broadcast-style virtual camera, virtual mic, and virtual speakers in one app.
- Daemon-first design keeps devices available while heavy processing wakes only when real apps start using them. Also this allows us to run headless... SPOOKY!
- Open backends first, optional Maxine power when you bring your own SDK.
- GPU-conscious pipelines built around efficiency, reuse, fewer transfers, and low-latency real-time calls with CPU fallback!
- Written in C++ with speed in mind!
- Don't like the ML models provided? First off, ouch. Secondly, we made StudioCast bring-your-own-model friendly so feel free to try almost any model you like.
- Safety-minded audio routing that avoids feeding StudioCast back into itself.
- Desktop GUI, CLI tools, diagnostics, installer flow, and service integration instead of a pile of one-off scripts.
- Built for OBS, Zoom, Teams, Discord, and browser calls, with the architecture to grow from hobby setup to daily-driver streaming rig.

<p align="center">
  <img src="docs/assets/studiocast-ui-preview.gif" alt="StudioCast UI preview" width="760">
</p>

## What it does

- Creates a virtual camera that can be selected in OBS, Zoom, Teams, Discord,
  and browser/WebRTC apps.
- Provides virtual audio devices for microphone and speaker routing.
- Runs optional video effects such as background blur/removal/replacement,
  denoise, relighting, eye contact, auto-framing, and mirroring when the required
  backend is available.
- Runs optional microphone and speaker effects such as noise removal and
  voice enhancement when the required backend is available.
- Uses a background daemon, `studiocastd`, so virtual devices can stay available
  while heavier processing starts only when an app is actually consuming them.

## Project status

StudioCast is an early-preview Linux project. It is usable for testing on
Ubuntu 22.04 and 24.04, with a source-build flow and an installer wizard target
for release packaging. Packaging, model installation, hardware behavior, and
some effects are still evolving.

Known caveats:

- The open-source eye-contact/eye-tracking path is best-effort and can be
  glitchy.
- Effect availability depends on local GPU drivers, ONNX Runtime, model packs,
  or an optional NVIDIA Maxine SDK installation.
- Development bandwidth may vary. Focused contributions and good bug reports are
  welcome.

## Requirements

- Ubuntu 22.04 or 24.04, or a close Ubuntu-family desktop distribution.
- A V4L2-compatible physical camera for camera input.
- `v4l2loopback` for the StudioCast virtual camera.
- A PulseAudio-compatible audio stack with `pactl` for virtual audio routing.
- CMake, Ninja, Qt, and compiler dependencies for the current source-build flow.
- Optional: NVIDIA driver/CUDA support for Open Video model backends.
- Optional: Vulkan loader/runtime packages plus a compute-capable GPU driver/ICD
  for Open Vulkan experiments.
- Optional: user-installed NVIDIA Maxine SDK assets for Maxine effects.

StudioCast does not ship NVIDIA Maxine SDK files or model binaries in git.

## Install options

- Download GUI installer: release packaging builds a versioned
  `StudioCast-Installer-<version>-<arch>.AppImage` when AppImage tooling is
  available, plus a staged AppDir archive, standalone source archive, and
  SHA256 checksums. Run the installer as your normal user, not with `sudo`. The
  AppImage/AppDir is self-contained for the installer GUI, scriptable backend,
  and matching StudioCast source archive; the GUI preselects that bundled
  archive for install/update workflows. Runtime dependencies still come from
  supported system packages and the backend setup scripts. The installer builds
  StudioCast on the target system; it is not a binary-only StudioCast
  application bundle.

- Build the GUI installer package locally:

```bash
packaging/appimage/build_appimage.sh
```

  If `linuxdeploy` and `linuxdeploy-plugin-qt` are installed, this also produces
  an AppImage. Without those tools, it produces a staged AppDir tarball and
  checksum for inspection/testing.

- Build the GUI installer target from source:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target studiocast-installer
./build/studiocast-installer
```

- Build from source manually: use the commands below when developing,
  installing over SSH, debugging setup, or recovering from a failed install.

The same backend used by the GUI is available as a CLI fallback:

```bash
./scripts/installer.sh status
./scripts/installer.sh plan install
./scripts/installer.sh install --yes
./scripts/installer.sh uninstall --yes
```

## Quick start from source

Run these commands from the repository root.

1. Install system dependencies and create the virtual camera:

```bash
./scripts/setup.sh --deps --v4l2loopback --load-loopback --persist-loopback
```

By default this creates a v4l2loopback device such as `/dev/video10` labeled
`StudioCast Camera`.

2. Build StudioCast:

```bash
./scripts/setup.sh --build --build-dir ./build --build-type Release
```

3. Start the daemon:

```bash
./build/studiocastd
```

4. In another terminal, check status and open the GUI:

```bash
./build/studiocastctl status --pretty
./build/studiocast
```

Optional: once the manual run works, install the systemd user service:

```bash
./scripts/install.sh user-service --build-dir ./build --yes
systemctl --user status studiocastd.service
```

More setup detail is available in [docs/SETUP.md](docs/SETUP.md).

## Using StudioCast

Start `studiocastd`, open the StudioCast GUI, then choose StudioCast devices in
the app you want to use.

- OBS: add a Video Capture Device and select `StudioCast Camera`.
- Zoom, Teams, Discord, and browser/WebRTC apps: select `StudioCast Camera` in
  camera settings. If the app already had its device list open, refresh or reopen
  the settings page.
- Audio apps: when StudioCast virtual audio devices are enabled, select
  `StudioCast Microphone` and/or `StudioCast Speakers` in the target app.
- GUI preview: enabling preview can act as a consumer of the virtual camera.
  Leave it off when you only want external apps to consume the output.

StudioCast is designed to idle when no consumer is using the virtual camera or
processed audio path. If it stays busy while no app is consuming StudioCast
devices, please file a bug with a support bundle.

## Models and effects

StudioCast can use multiple effect backends. The GUI and `studiocastctl status`
report which engines are available on the current machine.

- Open Video / Open CUDA: ONNX Runtime with CUDA execution provider and
  user-installed model packs. See
  [docs/open_source_video_models_install.md](docs/open_source_video_models_install.md).
- Open Vulkan: optional runtime-loaded Vulkan compute path. Build with
  `-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON`; explicit Vulkan requests report
  Vulkan fallback/degraded status instead of silently running CUDA.
- Open Audio: ONNX Runtime and user-installed model packs for microphone effects.
  See [docs/open_source_audio_models_install.md](docs/open_source_audio_models_install.md).
- NVIDIA Maxine: optional user-installed NVIDIA SDKs and feature packs. See
  [docs/maxine_install.md](docs/maxine_install.md).

Curated model installers are available through:

```bash
./scripts/install.sh open-video-models --list
./scripts/install.sh open-audio-models --list
```

Use the model helper to inspect installed packs:

```bash
./build/studiocast-open video-list-models
./build/studiocast-open audio-list-models
```

## Troubleshooting

Virtual camera does not appear:

```bash
v4l2-ctl --list-devices
./scripts/setup.sh --v4l2loopback --load-loopback --persist-loopback
```

Daemon is not reachable:

```bash
./build/studiocastd
./build/studiocastctl status --pretty
```

If using the user service:

```bash
systemctl --user status studiocastd.service
journalctl --user -u studiocastd.service -f
```

Apps show the StudioCast camera but no video:

- Confirm the daemon is running.
- Confirm a readable physical camera is selected or available for auto-select.
- Check `./build/studiocastctl status --pretty` for device, consumer, and
  pipeline state.
- Try OBS first when debugging, then browser/WebRTC apps.

Effects are unavailable:

- Check `./build/studiocastctl status --pretty` for engine diagnostics.
- For Open Video, verify NVIDIA driver/CUDA, ONNX Runtime, and model packs.
- For Open Audio, verify ONNX Runtime and model packs.
- For Maxine, run `./build/studiocast-maxine install-hints`.

Audio devices do not work:

- Check `pactl info`, `pactl list short sources`, and `pactl list short sinks`.
- In the GUI, select a physical microphone or speaker device rather than a
  StudioCast virtual device as the processing source/target.

Support bundle:

```bash
./build/studiocastctl debug-report --out studiocast-debug-report.txt
```

Attach the report when opening an issue.

## Contributing

Start with [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for build,
architecture, daemon, IPC, model, and testing notes.

Project conventions are in [CONTRIBUTING.md](CONTRIBUTING.md). Please also see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).

## License and trademarks

StudioCast is independent and is not affiliated with, endorsed by, or sponsored
by NVIDIA. NVIDIA and NVIDIA Broadcast are trademarks of NVIDIA Corporation.
StudioCast does not ship or redistribute NVIDIA Broadcast or Maxine SDK
binaries.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[docs/TRADEMARKS.md](docs/TRADEMARKS.md).
