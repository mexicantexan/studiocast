# StudioCast GUI Strengths Audit

- Status: Archived historical UX audit
- Applies To: GUI and screenshots reviewed on 2026-07-03
- Audience: Designers and maintainers investigating GUI history
- Archived: 2026-07-22

Date: 2026-07-03

Purpose: identify what is working well in the current StudioCast GUI and what
should be preserved, adapted, or respected during a full reface/reskin. This
report focuses on functional clarity, interaction patterns, affordances, safety
behaviors, and information architecture choices that support users. It does not
argue for preserving the current visual layout.

Historical evidence note: the predecessor functionality audit and four
2026-07-03 screenshots cited by the original report were not retained in this
repository. The observations below are preserved as historical context, not as
current product requirements. Current GUI authority is `docs/GUI_POLICY.md`;
current manual coverage is `docs/MANUAL_TESTING.md`.

## 1. Executive Summary

- The strongest current UX asset is that the GUI behaves as a daemon controller,
  with daemon reachability, active backend state, effect availability, and
  diagnostics treated as runtime facts rather than guessed client-side state.
- The three main device domains, Microphone, Speakers, and Camera, map well to
  how users must configure StudioCast in external apps such as OBS, Zoom,
  Discord, Teams, and browsers.
- The UI already exposes critical safety concepts: safe source/sink filtering,
  camera input/output conflict prevention, unavailable-effect disabling, missing
  model reporting, daemon-unavailable warnings, and preview-as-consumer behavior.
- The current app includes useful direct affordances for setup and recovery:
  Refresh controls, Auto/default device choices, copyable v4l2loopback modprobe
  guidance, install hints, explicit virtual-device lifecycle actions, and raw
  status details.
- The Camera page exposes a broad, coherent set of video effects with adjacent
  controls, including virtual background, auto frame, eye contact, denoise,
  virtual key light, and vignette.
- The audio pages make pass-through and virtual-device usage relatively clear:
  screenshots show Effect set to Off, strength controls, and tips telling users
  what StudioCast device to choose in other apps.
- Advanced and diagnostic functionality is present without being the only
  control surface: the audio pages have an Advanced affordance, and the Camera
  page has collapsible diagnostics/status.
- The current UI is visually dense and not a consumer-quality layout, but it
  contains many preservation-worthy behavior contracts that the redesign should
  carry forward in cleaner forms.

## 2. Preserve Or Adapt

### Daemon-First Control And Status

Current strength:

- The GUI uses the daemon as the source of truth for status, availability,
  backend diagnostics, disable reasons, model-pack diagnostics, and runtime
  errors.
- The Microphone and Speakers screenshots show a prominent daemon-unavailable
  warning with the failed socket path. The Camera page includes a video status
  area for daemon/device state.

Why it matters to users:

- Users need to know whether a problem is in the GUI, daemon, virtual device,
  backend engine, or model pack. Surfacing daemon state directly prevents false
  confidence and makes support/debugging possible.
- Polling also lets the UI recover when `studiocastd` starts, stops, restarts,
  or is changed through CLI/config outside the GUI.

Preserve, adapt, or hide:

- Preserve directly as an architectural rule.
- Adapt presentation from raw text-first diagnostics into concise health states
  with expandable details.

Historical functionality notes retained by this audit:

- `GET_STATUS` is used by audio and video pages.
- Camera polls every 500 ms; Microphone and Speakers poll every 1500 ms.
- Availability and disabled-state reasons must continue to come from daemon
  status.
- GUI daemon calls use UI-safe timeouts.
- Advanced/support access to raw daemon status and diagnostics is a reskin
  requirement.

### Device-Domain Information Architecture

Current strength:

- The current shell is organized around Microphone, Speakers, and Camera, which
  match the three virtual-device categories users must understand.
- The screenshots show these domains as top-level tabs, with page headings that
  reinforce the current task context.

Why it matters to users:

- StudioCast is used by selecting virtual camera, microphone, and speaker
  devices in other apps. Keeping these concepts explicit helps users connect GUI
  settings to what they choose in OBS, Zoom, Teams, Discord, and WebRTC.

Preserve, adapt, or hide:

- Adapt. The exact tab layout does not need to remain, but the three device
  concepts should remain first-class and easy to distinguish.

Historical functionality notes retained by this audit:

- Main window has three pages: Microphone, Speakers, Camera.
- README positions StudioCast around a virtual camera, virtual microphone, and
  virtual speakers.
- Audio and camera pages configure distinct device pipelines and virtual-device
  behavior.

### External-App Usage Guidance

Current strength:

- The Microphone screenshot includes a tip: set Effect to Off for pass-through
  and choose `StudioCast Microphone` in other apps.
- The Speakers screenshot says speaker effects apply through `StudioCast
  Speakers` and tells users to select `StudioCast Speakers` as the output device.
- The Camera page labels the flow as processed camera to virtual camera.

Why it matters to users:

- The app's success depends on users picking the right virtual device elsewhere.
  Embedded guidance reduces the gap between configuring StudioCast and making it
  work in a call, stream, or recording app.

Preserve, adapt, or hide:

- Preserve directly, but adapt the copy into more polished, contextual guidance.
- Keep exact virtual-device names visible wherever they are needed.

Historical functionality notes retained by this audit:

- Microphone and Speakers pages include pass-through/effects tips.
- README instructs users to select StudioCast virtual camera, microphone, and
  speakers in target apps.
- Speaker controls route audio through `StudioCast Speakers`.

### Backend Preference Versus Active Backend

Current strength:

- Audio and Camera pages distinguish the selected backend preference from the
  active backend. The screenshots show `Backend: Auto` and `Active: -` as
  separate values.

Why it matters to users:

- `Auto` can resolve to different engines, pass-through, mixed state, or no
  active engine depending on installed runtimes, models, and enabled effects.
  Showing both preference and actual state prevents users from assuming a
  backend is active just because they selected it.

Preserve, adapt, or hide:

- Preserve conceptually and make the distinction more visually legible.
- Do not collapse preference and active runtime state into one label.

Historical functionality notes retained by this audit:

- Microphone and Speakers support backend preference choices: Auto, Maxine, Open
  Source, Off.
- Camera supports backend preference choices: Auto, Maxine, Open Source.
- Active backend labels are derived from daemon status and can show pass-through,
  starting, off, mixed, or backend-specific states.
- Backend notes and fallback explanations come from daemon status.

### Warning And Info Banners

Current strength:

- The Microphone and Speakers screenshots show a highly visible warning banner
  when the daemon socket is unavailable.
- The audit describes info banners for daemon effect notes/fallback notes and
  warning banners for daemon/unavailable errors.

Why it matters to users:

- Banners keep important state near the controls affected by that state. They
  also avoid silent failure when a backend, daemon, model, or device cannot be
  used.

Preserve, adapt, or hide:

- Preserve directly.
- Adapt language so the default message is plain-language and the technical
  daemon text is available as details.

Historical functionality notes retained by this audit:

- Audio pages display daemon notes/fallback notes and unavailable errors.
- Camera displays engine-blocking warnings and backend notes.
- Blocking failures use critical dialogs, while contextual issues use banners or
  status text.

### Safe Device Selection And Recovery Controls

Current strength:

- Audio pages expose clear input/output selectors with Refresh buttons.
- The Camera page exposes input camera, output v4l2loopback, width, height, FPS,
  Refresh, and Copy modprobe command controls.
- Device labels include concrete Linux identifiers such as `/dev/video10`,
  PulseAudio sink names, and `[loopback]` markers.

Why it matters to users:

- Users often attach/detach devices, change defaults, or run without a daemon or
  loopback module. Auto/default selections and Refresh controls help recover
  without restarting the app.
- Filtering unsafe devices reduces the chance of routing StudioCast output back
  into itself or selecting monitor/loopback devices incorrectly.

Preserve, adapt, or hide:

- Preserve safety filtering and refresh/recovery directly.
- Adapt low-level labels with friendlier names while keeping technical details
  available for support and power users.
- Keep v4l2loopback setup guidance accessible, likely under setup/support UI.

Historical functionality notes retained by this audit:

- Audio source/sink lists use `pactl`.
- Unsafe input and speaker candidates are filtered with audio device safety
  helpers.
- Missing configured sources are shown as `<missing: ...>`; disconnected
  speaker sinks are shown as `<disconnected: ...>`.
- Camera device labels include node, name, driver, and `[loopback]`.
- Output shows `<auto>` when writable loopback exists and a disabled placeholder
  when no writable loopback is found.
- Copy modprobe command copies the suggested v4l2loopback command.

### Camera Preview As An Explicit Verification Surface

Current strength:

- The Camera top screenshot gives the preview a large, visible area with a clear
  `Preview off` state and an explicit Preview checkbox.

Why it matters to users:

- A live preview is the simplest confidence check before joining a call or
  stream. An explicit off state also avoids confusion about whether the camera
  should currently be active.

Preserve, adapt, or hide:

- Preserve directly and make it a first-class camera test/verification affordance.
- Keep the preview-consumer consequence visible near the preview control.

Historical functionality notes retained by this audit:

- `VideoPreviewWidget` displays status messages or rendered frames.
- Preview starts only when daemon video is enabled.
- Preview opens the virtual output camera, uses the daemon-resolved output
  device/format when available, and supports RGB24/YUYV capture.
- Preview counts as a camera consumer and can trigger the consumer-gated daemon
  pipeline.
- Preview failures use bounded backoff; stopping preview releases the capture
  handle and clears buffers.

### Consumer-Gated Processing Transparency

Current strength:

- The architecture and status model explain that heavy processing runs only when
  a consumer is using the virtual devices.
- Camera status includes messaging that preview counts as a consumer. Speaker
  status distinguishes pass-through loopback routing from processed pipeline
  routing.

Why it matters to users:

- Users may otherwise interpret idle processing as broken, or active processing
  during preview as unexpected resource use. Clear consumer-gated messaging
  protects trust around CPU/GPU load and device behavior.

Preserve, adapt, or hide:

- Preserve directly as a normal status concept, not only as raw diagnostics.

Historical functionality notes retained by this audit:

- README states heavy processing should avoid running when no app consumes
  virtual devices.
- Camera status reports consumer count, pipeline state, idle reason, and preview
  consumer behavior.
- Speaker status reports consumer counts, route mode, routing active state, and
  processed pipeline state.

### Direct Effect Controls With Visible Values

Current strength:

- Effects are represented with concrete controls: Off/effect selectors, Enable
  toggles, sliders, spin boxes, file pickers, and adjacent numeric values.
- The screenshots show common defaults and current values clearly, such as
  Effect Off, Strength 50, camera Blur strength 50%, Eye Contact strength 50%,
  Key Light intensity 70%, and Vignette intensity 35%.

Why it matters to users:

- Users can see what is active, what is disabled, and what parameter value will
  be applied. Off/pass-through states are explicit, which makes experimentation
  easier to reverse.

Preserve, adapt, or hide:

- Preserve the direct-manipulation model.
- Adapt the layout and visual controls to reduce density and group effects by
  user intent, while retaining clear current values and off states.

Historical functionality notes retained by this audit:

- Microphone effects: Off, Noise removal, Room echo removal, Noise + echo
  combined, Studio voice, plus strength.
- Speaker effects: Off, Noise removal, Room echo removal, Noise + echo combined,
  plus strength.
- Camera effects: Virtual Background, Auto Frame, Eye Contact, Video Noise
  Removal, Virtual Key Light, and Vignette.
- Strength controls are disabled when the corresponding effect is Off where
  applicable.
- Camera and audio effect changes write through to daemon config.

### Write-Through Persistence

Current strength:

- Most controls send config changes immediately after user interaction and the
  daemon persists successful changes.

Why it matters to users:

- The UI has a simple mental model: changing a control changes the running or
  saved configuration. Users are less likely to lose changes because they forgot
  an Apply button.

Preserve, adapt, or hide:

- Preserve unless the redesign intentionally adopts a visible Apply/Cancel
  model.
- If changed, the new behavior must be explicit because it alters a core
  interaction contract.

Historical functionality notes retained by this audit:

- Audio controls send `SET_AUDIO_CONFIG`.
- Camera effect controls send `SET_VIDEO_EFFECTS_JSON`.
- Daemon persists config changes after successful `SET_*` commands.
- Reskin requirements explicitly call for preserving write-through persistence
  or intentionally replacing it with an explicit Apply/Cancel model.

### Camera Start And Stop Guardrails

Current strength:

- Camera start is staged: send device/format config, send effects JSON, then
  enable daemon video.
- Camera stop closes preview and disables daemon video.
- Input/output/format controls are disabled while video is enabled.

Why it matters to users:

- The sequence prevents partial or inconsistent camera activation. Locking setup
  fields while the camera is running also reduces accidental disruption during a
  call or recording.

Preserve, adapt, or hide:

- Preserve directly.
- The new UI can change the shape of the controls, but should keep the same
  operational safety.

Historical functionality notes retained by this audit:

- Start sends `SET_VIDEO_CONFIG`, `SET_VIDEO_EFFECTS_JSON`, and then
  `SET_ENABLED enabled=1`.
- Stop closes preview and sends `SET_ENABLED enabled=0`.
- Start prevents using the same device as input and output.
- If input is explicit and output is auto, the GUI tries to resolve a different
  writable output.
- Input/output/format controls are disabled while camera video is enabled.

### Model Availability And Missing Model Reporting

Current strength:

- The GUI preserves missing configured model IDs instead of silently falling
  back to auto.
- Model controls appear in relevant contexts, and install hints are available
  for Open Source and Maxine.

Why it matters to users:

- Silent fallback would make effects appear configured while actually using a
  different model or no model. Explicit missing/unavailable state supports
  repair and reliable bug reports.

Preserve, adapt, or hide:

- Preserve directly as a trust behavior.
- Adapt detailed model selection and install hints into a cleaner model/backend
  management or diagnostics surface.

Historical functionality notes retained by this audit:

- Audio model selectors include `<auto>`, installed models, unavailable states,
  and `<missing: ...>` configured IDs.
- Optional explicit `.onnx` paths are supported for audio.
- Camera model rows are shown when relevant effects are enabled and Open Source
  is selected/effective.
- Diagnostics install hints run `studiocast-open install-hints`,
  `studiocast-maxine install-hints`, or both in Auto mode.
- Unavailable effect modes/items are disabled where possible.

### Virtual Device Lifecycle Actions

Current strength:

- The Speakers screenshot exposes explicit actions: Enable speakers device,
  Stop routing, and Destroy speakers device. Destructive action is visually
  distinguished.
- The audit documents virtual microphone create/destroy and daemon-managed
  speaker lifecycle controls.

Why it matters to users:

- Virtual devices are system-facing objects. Users need a way to create, stop,
  and remove them, but those actions should be deliberate and visibly different
  from everyday effect tuning.

Preserve, adapt, or hide:

- Preserve the functionality.
- Hide or de-emphasize these controls under Advanced/System/Device Management
  for a wide-audience UI, while keeping status and repair paths easy to find.

Historical functionality notes retained by this audit:

- Virtual mic create/destroy is daemon-managed via `SET_AUDIO_CONFIG`.
- Speakers can enable the virtual speakers device, stop routing, and destroy the
  virtual speakers device.
- Release builds show an error rather than using direct fallback mutation when
  the daemon is unavailable.
- Debug builds may use direct `pactl` helper fallbacks.

### Diagnostics And Support Escape Hatch

Current strength:

- Advanced/status and diagnostics areas exist for users who need raw state.
- The Camera page has a checkable/collapsible Diagnostics group and a video
  status area. Audio pages expose Advanced status.
- Help -> About includes version, git SHA, short app description, and NVIDIA
  non-affiliation notice.

Why it matters to users:

- StudioCast depends on Linux device state, a daemon, model packs, and optional
  backends. Support-grade detail is necessary even if the default UI becomes
  much simpler.

Preserve, adapt, or hide:

- Preserve as Advanced/Diagnostics/Support UI.
- Do not make raw JSON/status the default experience, but keep it copyable and
  complete.

Historical functionality notes retained by this audit:

- Microphone status combines local audio status with daemon audio status.
- Speaker status includes presence, consumers, routing, target sink, pipeline
  state, backend, timing, overruns, processed frames, and errors.
- Camera status includes v4l2loopback probe text, daemon reachability, device
  availability, consumer count, pipeline state, formats, effect summary,
  backend summaries, notes, frame index, and last error.
- Diagnostics show engine-specific daemon JSON and missing model notes.

## 3. User Trust And Safety Strengths

- Daemon-unavailable state is visible near affected controls, as shown on the
  Microphone and Speakers screenshots.
- Runtime availability comes from daemon status instead of client-side guesses.
- Unavailable effects can be disabled, and daemon-supplied disabled reasons can
  be exposed through tooltips or diagnostics.
- Existing enabled effects may remain selectable so users can turn them off even
  if the backing engine later becomes unavailable.
- Audio source/sink safety helpers filter unsafe StudioCast, monitor, and
  loopback-style device candidates.
- Missing or disconnected configured devices are preserved in the UI rather than
  silently replaced.
- Missing configured model IDs are preserved and shown as missing instead of
  silently falling back to auto.
- Camera start prevents using the same device as input and output.
- Camera setup controls are disabled while video is enabled.
- Preview is explicit, starts only when daemon video is enabled, and is reported
  as a consumer.
- Preview open failures use bounded backoff rather than retrying aggressively.
- Stopping preview releases the V4L2 capture handle and clears frame buffers.
- The daemon keeps the virtual camera discoverable but does not try to load or
  create the kernel module; the GUI offers/copies the suggested modprobe command
  instead.
- Release builds do not perform direct legacy audio mutations when the daemon is
  unavailable.
- Virtual speakers lifecycle controls distinguish enable, stop routing, and
  destroy actions.
- Raw status and diagnostics remain available for support without requiring
  users to infer hidden state.

## 4. Functional Completeness Strengths

- Main application structure covers the product's core device categories:
  Microphone, Speakers, and Camera.
- Microphone functionality exposes backend preference, active backend, daemon
  banners, Pulse source selection, effect selection, strength, Open Audio model
  selection/path, install hints, virtual microphone lifecycle, legacy loopback
  controls, and advanced status.
- Speakers functionality exposes backend preference, active backend, target sink
  selection, speaker effects, strength, Open Audio model selection/path, virtual
  speakers enable/stop/destroy controls, routing tips, and detailed status.
- Camera functionality exposes preview, device refresh, input camera selection,
  v4l2loopback output selection, modprobe command access, width/height/FPS,
  backend preference, active backend, start/stop, video effects, model
  diagnostics, install hints, and detailed status.
- Video effect coverage is broad enough to represent StudioCast's differentiator
  capabilities rather than only basic setup.
- Audio effect coverage exposes practical cleanup modes for microphone and
  speakers, with Off/pass-through as a visible baseline.
- The GUI already bridges runtime controls with setup/troubleshooting surfaces:
  daemon status, install hints, model diagnostics, device refresh, and virtual
  device lifecycle controls.
- Write-through control behavior and polling make the GUI a live controller for
  `studiocastd`, not a static preferences editor.

## 5. Redesign Implications

- Keep Microphone, Speakers, and Camera as first-class user concepts even if the
  navigation and page structure change.
- Keep daemon status and daemon-reported availability authoritative; summarize
  it by default and preserve raw details for support.
- Preserve safety guardrails independently of visual design: unsafe audio
  filtering, camera input/output conflict prevention, disabled unavailable
  effects, missing model reporting, and daemon-unavailable handling.
- Keep exact external-app device names visible at the point where users need to
  configure another app.
- Preserve the explicit preview/test model and clearly explain when preview or
  monitoring counts as a consumer.
- Preserve pass-through/off states as clear, reversible defaults.
- Keep backend preference separate from active backend state.
- Treat virtual device lifecycle, raw diagnostics, direct legacy controls, and
  model-pack repair as advanced/support capabilities rather than everyday
  effects controls.
- If the redesign changes write-through persistence, make the new Apply/Cancel
  behavior explicit and consistent.
