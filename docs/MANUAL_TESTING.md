# Manual Testing

Status: manual regression plan for the refaced GUI and hardware desktop
workflows.
Last updated: 2026-07-03.

This document tracks manual tests that cannot be covered deterministically in
CI because they require real V4L2 devices, v4l2loopback behavior, desktop
consumers, GPU runtimes, or user permissions.

## Scope

Primary scope:

- GUI control of the daemon-driven video pipeline.
- Physical camera input selection.
- v4l2loopback output selection and consumer detection.
- GUI preview behavior.
- Daemon IPC failure and recovery behavior.
- Refaced GUI navigation: Home, Camera, Microphone, Speakers, Engines &
  Models, Support, Settings, and Advanced.
- Effect availability as reported by the daemon.
- Raw diagnostics access from Support and Advanced.
- Advanced lifecycle, legacy/debug, and destructive-confirmation behavior.
- Config persistence across GUI and daemon restarts.

Out of scope for this first pass:

- Full model quality assessment.
- Long-duration soak testing.
- Packaging on every supported distro.
- Performance tuning beyond obvious hangs, tight loops, or runaway CPU.

## Test Record

Create one completed record per test pass.

```text
Date:
Tester:
Git commit:
Build type:
OS and kernel:
Desktop session:
GPU and driver:
Camera devices:
v4l2loopback version:
Daemon launch mode: manual / systemd user service
Consumer apps tested: OBS / browser / Zoom / Teams / Discord / other
Result: pass / fail / partial
Notes:
Artifacts:
```

Useful artifacts:

- `build/studiocastctl status`
- `build/studiocastctl debug-report --out studiocast-debug-report.txt`
- `pactl info`
- `pactl list short sources`
- `pactl list short sinks`
- `pactl list short modules`
- `v4l2-ctl --list-devices`
- `v4l2-ctl --all -d /dev/videoX`
- GUI screenshots for user-visible errors or disabled controls
- Daemon logs from the terminal or `journalctl --user -u studiocastd.service`

## Prerequisites

- Ubuntu 22.04 or 24.04 desktop session.
- StudioCast built:

```bash
cmake --build build --target studiocast studiocastd studiocastctl studiocast-gui-status-tests
```

- v4l2loopback installed and loaded.

```bash
./scripts/setup.sh --v4l2loopback --load-loopback
v4l2-ctl --list-devices
```

- At least one readable physical camera for physical input tests.
- At least one physical microphone/input source and one physical speaker/output
  sink for audio routing tests.
- A desktop consumer app such as OBS or a browser/WebRTC camera test page.
- Optional: a second readable loopback device for loopback-as-input tests.
- Optional: Maxine SDK or Open CUDA model packs for positive effect availability
  tests.

## Baseline Sanity

- [ ] Run `build/studiocastctl status` before starting the daemon.

Expected:

- Command returns quickly.
- Error is clear that the daemon socket is missing or unreachable.
- It does not hang.

- [ ] Start the daemon manually:

```bash
build/studiocastd
```

- [ ] Run:

```bash
build/studiocastctl status
v4l2-ctl --list-devices
```

Expected:

- Status reports the configured StudioCast virtual camera.
- The virtual device exists in `v4l2-ctl --list-devices`.
- With no consumer open, heavy video processing is not running.

## GUI And Daemon IPC

- [ ] Start the GUI with no daemon running.

Expected:

- GUI remains responsive.
- The Home page reports daemon unavailable instead of showing a healthy setup
  checklist.
- The header service status reports the daemon is unavailable.
- Start/effects controls that require the daemon are disabled or fail with an
  actionable message.
- Support and Advanced still show copyable raw diagnostics with the transport
  error.
- Preview stays off.

- [ ] Start the daemon while the GUI is already open.

Expected:

- GUI recovers without restart on the next poll or refresh.
- Header and Home status switch from unreachable to reachable.
- Device and effect state match daemon status.

- [ ] Kill the daemon while the GUI is open, then restart it.

Expected:

- GUI does not hang.
- Controls reflect daemon unreachable while it is down.
- Preview closes or reports unavailable.
- After daemon restart, controls and status recover without restarting the GUI.

- [ ] Change camera settings in the GUI, close the GUI, restart it, and compare:

```bash
build/studiocastctl status
```

Expected:

- GUI state, daemon status, and persisted config agree.
- Failed daemon/config writes are not shown as successful GUI state.

## Refaced GUI Navigation And Status

- [ ] Open each top-level page from the sidebar:
  Home, Camera, Microphone, Speakers, Engines & Models, Support, Settings, and
  Advanced.

Expected:

- Sidebar selection changes the page without resetting device state.
- Header service status remains visible and continues polling while navigating.
- Camera preview remains off unless the Camera page preview checkbox is
  explicitly enabled.
- Settings reset actions appear only on Settings.
- System/debug lifecycle controls appear only in Advanced or in demoted
  device-page details.

- [ ] With daemon healthy and no current blockers, inspect Home.

Expected:

- Home shows Camera, Microphone, and Speakers readiness cards.
- Home shows the external app device names:
  `StudioCast Camera`, `StudioCast Microphone`, and `StudioCast Speakers`.
- Copy buttons put the exact device names on the clipboard.
- Healthy Home does not show a setup checklist or repair queue.

- [ ] Introduce or simulate a blocker, such as stopping the daemon, unloading
  v4l2loopback, selecting a missing model ID, or disconnecting a configured
  physical source/sink.

Expected:

- Home shows a repair/setup queue only while a blocker exists.
- Repair items link to the relevant page: Camera, Microphone, Speakers,
  Engines & Models, or Support.
- The issue summary matches daemon status and does not rely on stale GUI state.

- [ ] Open Engines & Models with the daemon running.

Expected:

- Camera and audio backend preference are distinct from active backend state.
- Maxine, Open Video, and Open Audio health comes from daemon diagnostics.
- Installed model packs, missing model packs, blocked effects, configured
  missing model IDs, explicit model paths, and install hints are visible when
  reported by the daemon.
- Open Video/Open Audio cards expose a download button when default or missing
  model packs are needed, run the model installer without freezing the GUI, and
  refresh diagnostics after completion.
- Raw engine diagnostics remain visible and copyable from the page text boxes.

- [ ] Open Support.

Expected:

- Plain-language issue summaries are shown above technical details.
- Raw daemon status remains visible in the Raw Daemon Status box.
- Copy Issue Details, Copy Install Hints, and Copy Raw Status buttons work.
- Generate Support Report runs `studiocastctl debug-report --out ...` and shows
  a clear failure if the tool cannot be started.
- Version, git SHA, and service state are visible.

- [ ] Open Advanced.

Expected:

- Socket path and raw daemon JSON/status are visible and copyable.
- Explicit Open Video raw model IDs and Open Audio raw model IDs/paths remain
  editable through existing daemon config writes.
- Always-on camera reflects daemon status and writes with
  `SET_VIDEO_CONFIG always_on=0|1`.
- Advanced does not hide Support raw diagnostics; both pages provide a raw
  status path for support.

## Settings And Reset Actions

- [ ] Use each Settings scoped reset independently:
  reset camera effects, reset microphone effects, reset speaker effects, reset
  device selections to Auto, and restore all defaults.

Expected:

- Each reset asks for confirmation before persistent write-through changes.
- Scoped resets only modify the named surface.
- Restore all defaults requires confirmation and reports each write step.
- After each reset, GUI state and `build/studiocastctl status` agree.
- Failed writes leave the previous daemon state visible rather than claiming a
  reset succeeded.

## Device Selection

- [ ] With one physical camera and one StudioCast loopback output, open the GUI.

Expected:

- Physical camera is offered as an input.
- Writable StudioCast loopback is offered as output.
- The same `/dev/videoX` cannot be selected as both input and output.
- Labels make loopback devices distinguishable from physical devices.

- [ ] Select automatic input and automatic output, then start the camera.

Expected:

- Auto-selection chooses a readable input and a separate writable output.
- If no safe separate pair exists, start fails with an actionable error.

- [ ] If a readable loopback input device is available, select it as input and
  the StudioCast output as output.

Expected:

- Loopback-as-input works only when it is not the same device as the output.
- The GUI prevents input/output conflicts.
- Daemon status reports the resolved input and output devices clearly.

- [ ] Remove or block access to the selected input camera, then refresh/start.

Expected:

- Missing or permission-denied device states are visible.
- User can choose another device and recover without restarting the app.

## Consumer-Gated Pipeline Lifecycle

- [ ] Start the daemon and enable video with no OBS/browser/preview consumer.

Expected:

- Virtual output device is present and available.
- Pipeline state is idle because no consumer is present.
- CPU usage remains low.

- [ ] Open OBS or another consumer and select the StudioCast virtual camera.

Expected:

- Daemon detects a consumer.
- Pipeline starts after the configured grace period.
- GUI status shows consumer count and running/starting state.
- Consumer receives frames.

- [ ] Close the consumer.

Expected:

- Daemon detects the consumer is gone.
- Pipeline stops after the configured stop grace/min-run window.
- GUI status reflects idle/no-consumer state.

- [ ] While a consumer is active, change input device, output device,
  resolution, FPS, and effects one at a time.

Expected:

- Each change either succeeds and is reflected in daemon status, or fails with
  an actionable error.
- Restarts are bounded.
- No tight retry loop, runaway CPU, or GUI freeze occurs.
- Consumer recovers frames after each successful restart.

## GUI Preview

- [ ] Open the GUI with preview unchecked and video enabled.

Expected:

- GUI preview does not open the virtual camera.
- Preview does not count as a consumer until requested.

- [ ] Check the Preview box.

Expected:

- Preview opens the daemon-resolved virtual output device.
- Daemon consumer count increases.
- Pipeline starts if no other consumer is present.
- Preview shows processed output frames, not the raw physical camera feed.

- [ ] Uncheck the Preview box.

Expected:

- Preview stops.
- File descriptors are released.
- If no other consumer remains, pipeline returns to idle after the grace period.

Optional evidence:

```bash
lsof /dev/videoX
build/studiocastctl status
```

- [ ] Toggle preview repeatedly while starting/stopping OBS.

Expected:

- Retries are bounded.
- Status remains understandable during exclusive-caps transitions.
- No persistent stuck preview state remains after consumers close.

## Audio Device And Routing

- [ ] Open the Microphone page with the daemon running and inspect the input
  selector.

Expected:

- Physical microphone/input sources are selectable.
- StudioCast virtual sources and Pulse monitor sources are absent from the
  selectable list.
- `Auto (Pulse default)` is available.
- A persisted missing source appears as missing/disconnected instead of being
  silently replaced.

- [ ] Select a physical microphone source, enable a microphone effect or
  pass-through mode, then check:

```bash
build/studiocastctl status
```

Expected:

- `audio.source` matches the selected source or `auto`.
- `audio.source_resolved` is a physical microphone/input source.
- If the daemon rejects the change, the GUI reports the error and does not leave
  the rejected source shown as applied.

- [ ] Set the Pulse default source to a monitor source, then set StudioCast
  microphone input to `Auto`.

Expected:

- `SET_AUDIO_CONFIG` rejects the config when no safe source exists, or status
  reports that a safe physical source was selected instead.
- The daemon never captures from `studiocast_mic`,
  `studiocast_speakers.monitor`, or another sink monitor for the microphone
  pipeline.
- The error tells the tester to choose a physical microphone/input source.

- [ ] Open the Speakers page and inspect the output selector.

Expected:

- Physical output sinks are selectable.
- StudioCast virtual sinks and monitor devices are not offered as normal
  targets.
- A persisted missing target sink appears as disconnected/unavailable and
  remains visible until another sink is selected.

- [ ] Select a physical output sink, enable StudioCast Speakers with speaker
  effects off, and play audio from an app into `StudioCast Speakers`.

Expected:

- Daemon status reports `audio.speakers.route_mode=loopback`.
- `audio.speakers.target_sink` matches the selected sink or `auto`.
- `audio.speakers.target_sink_active` is the resolved physical sink.
- Audio reaches the selected physical speakers.

- [ ] Enable a speaker effect and keep the same physical output sink selected.

Expected:

- Daemon status switches to `audio.speakers.route_mode=pipeline`.
- The Pulse module-loopback route is stopped before processed speaker routing
  starts.
- `target_sink_active` remains the selected physical sink while the pipeline is
  running.
- If the pipeline cannot start, the GUI/status show the failure and do not claim
  routing is active.

- [ ] Kill and restart `studiocastd` while the GUI is open and audio devices are
  enabled.

Expected:

- GUI controls become unavailable while IPC is down.
- Release builds do not create/destroy virtual audio devices directly via
  `pactl` while the daemon is unreachable.
- After daemon restart, microphone source, speaker target, effect state, and
  routing state resync from daemon status.

- [ ] Create stale StudioCast Pulse modules manually or by killing the daemon
  mid-route, then restart the daemon and use GUI stop/destroy actions.

Expected:

- Stale `module-loopback`, `module-null-sink`, and `module-remap-source`
  instances are cleaned up or reported with actionable errors.
- Failed cleanup preserves the old active route in status instead of claiming a
  new route is active.

- [ ] Unplug or disable the selected microphone/source or selected speaker sink
  while routing or processing is active.

Expected:

- Status reports disconnected/unavailable source or target state.
- The GUI preserves the missing persisted selection visibly.
- Selecting another physical source/sink recovers without restarting the GUI.

## Advanced Lifecycle And Debug Controls

- [ ] In Advanced, create the StudioCast virtual microphone with the daemon
  reachable.

Expected:

- The action writes through the daemon using the existing audio config command.
- Advanced lifecycle status and `build/studiocastctl status` report the
  configured/present virtual microphone state.

- [ ] In Advanced, destroy the StudioCast virtual microphone.

Expected:

- A destructive confirmation dialog appears before the daemon write.
- Cancelling the dialog leaves daemon status unchanged.
- Confirming sends the existing daemon-managed destroy/disable config.
- In debug builds only, direct `pactl` fallback may be used if the daemon is
  unreachable.
- In release builds, daemon-unreachable direct virtual microphone mutation is
  blocked with an error.

- [ ] In Advanced, enable, stop, and destroy StudioCast Speakers.

Expected:

- Enable is non-destructive and writes the existing daemon audio config fields.
- Stop routing leaves the virtual speakers device configured/present when the
  daemon succeeds.
- Destroy shows a destructive confirmation dialog.
- Cancelling destroy leaves daemon status unchanged.
- Confirming destroy sends the existing daemon-managed
  `speakers_enabled=false` and `create_virtual_speakers=false` config.
- In debug builds only, direct `pactl` fallback may be used if the daemon is
  unreachable.
- In release builds, daemon-unreachable direct virtual speakers mutation is
  blocked with an error.

- [ ] Inspect the Legacy Loopback / Debug group in Advanced.

Expected:

- Debug builds allow the existing direct module-loopback start/stop controls
  when `pactl` is available.
- Release builds label the group as disabled and hide or disable mutation
  buttons.
- Local PulseAudio status remains readable so stale modules can be diagnosed.
- These controls are clearly separate from the daemon-managed audio pipeline.

## Effects Availability

- [ ] Run on a machine without Maxine SDK and without Open CUDA model packs.

Expected:

- GUI uses daemon-reported availability.
- Unsupported effect controls are disabled or marked unavailable.
- User-visible reason/hints explain missing runtime or models.
- Saved effect intent is not silently discarded.

- [ ] If Open CUDA model packs are installed, select Open CUDA and enable a
  supported background effect.

Expected:

- GUI lists daemon-reported installed models.
- Missing models are reported as missing, not guessed client-side.
- Pipeline starts with the selected effect when a consumer is present.

- [ ] If Maxine SDK is installed, select Maxine and enable supported effects.

Expected:

- GUI reflects daemon-reported Maxine support.
- Blocked effects show accurate reasons.
- Unsupported controls cannot create a mismatched GUI/backend state.

- [ ] Change effects while the pipeline is running.

Expected:

- Updates are applied atomically enough that GUI state, daemon config, and
  resulting video output stay consistent.
- Failure leaves the previous working daemon state intact or clearly reports
  what changed.

## Open Vulkan Adapter Selection

- [ ] On a system with multiple Vulkan devices, inspect
  `engines.open_vulkan.device_candidates` in daemon status.

Expected:

- Every loader-enumerated device has a stable `enumeration_index` for that run,
  device/vendor/type details, its compute queue family or a rejection reason,
  and selected/eligible state.
- Automatic selection prefers a discrete GPU, then an integrated GPU, then a
  virtual GPU. A CPU Vulkan implementation such as lavapipe is not treated as
  hardware acceleration by default.
- `selected_device_index`, `device_selection_request`, and the top-level device
  fields agree with the selected candidate.

- [ ] Select a specific candidate and restart the daemon:

```bash
STUDIOCAST_VULKAN_DEVICE_INDEX=1 build/studiocastd
build/studiocastctl status --pretty
```

Expected:

- Open Vulkan selects enumeration index `1`, or reports
  `vulkan_requested_device_not_found` / `vulkan_requested_device_no_compute_queue`
  without silently choosing another adapter.
- Vulkan enumeration indices are not assumed to match NVIDIA `nvidia-smi`
  indices or the generic `STUDIOCAST_GPU_INDEX` setting.

- [ ] On a CPU-only Vulkan system, test the explicit software-device opt-in:

```bash
STUDIOCAST_VULKAN_ALLOW_CPU=1 build/studiocastd
build/studiocastctl status --pretty
```

Expected:

- Without the opt-in, status reports `vulkan_only_cpu_devices_available` and
  explains that a working GPU driver/ICD is needed.
- With the opt-in, the CPU candidate can initialize, but diagnostics set
  `cpu_device_selected=true` and warn that it is software fallback rather than
  hardware GPU acceleration.

## Error And Recovery Cases

- [ ] Start GUI and daemon, then unplug the physical camera during streaming.

Expected:

- User-visible error is actionable.
- Daemon and GUI remain responsive.
- Reconnecting or selecting another input recovers without app restart.

- [ ] Remove the v4l2loopback module or make the output device unavailable.

Expected:

- Virtual device missing/unavailable state is visible.
- Start fails without claiming success.
- Reloading v4l2loopback and refreshing allows recovery.

- [ ] Run with insufficient device permissions.

Expected:

- Permission errors are surfaced as permission problems.
- Suggested fix is understandable to a non-developer tester.

- [ ] Generate a support bundle after a failure:

```bash
build/studiocastctl debug-report --out studiocast-debug-report.txt
```

Expected:

- Debug report includes enough device, daemon, status, and config information
  to diagnose the failure.

## Pass Criteria For Production Readiness

- No GUI hangs during daemon unreachable, stale socket, daemon restart, or
  malformed/failing command paths.
- No heavy video processing when no consumer is present and preview is off.
- Preview acts as an explicit consumer only when enabled by the user.
- Input/output device conflicts are prevented.
- Status text accurately reflects daemon state and is useful to non-developers.
- Persisted config matches GUI state after GUI and daemon restart.
- Effects availability comes from daemon diagnostics.
- Recoverable failures can be recovered from the UI without restarting the app.
- Support artifacts are sufficient to diagnose failures.

## Known Manual Gaps

- Exclusive-caps state transitions need validation on real v4l2loopback devices.
- Browser/WebRTC behavior may differ from OBS and should be tested separately.
- GPU effect availability needs separate passes for no GPU, unsupported GPU,
  Open CUDA installed, and Maxine installed.
- Long-running stability and thermal/performance behavior need a separate soak
  plan.
