# StudioCast Model Policy

This document defines how StudioCast should support user-supplied and curated
open-source model packs. It is the stance reviewers should use when evaluating
model-pack schemas, installer changes, GUI model selection, daemon diagnostics,
and runtime model loading.

The short version: StudioCast supports a documented local model-pack contract.
The daemon is the runtime authority, model binaries are user-installed or
intentionally curated, and model discovery must not become live-path work.

## Current Support Map

StudioCast has two open model families today.

Open Audio:

- Runtime: ONNX Runtime through the `open_audio` backend.
- Root: `${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_audio/<model_id>/`.
- Pack shape: `model.json`, an ONNX file referenced by `onnx_filename`, and
  `LICENSE.txt`.
- Selection fields: `microphone.model_id`, `microphone.model_path`,
  `speaker.model_id`, and `speaker.model_path` in the canonical audio effects
  schema.
- Resolution order: explicit `model_path`, then installed `model_id`, then a
  deterministic default installed pack for the enabled effect.
- Direct user paths are supported for advanced audio use: `model_path` may point
  to an ONNX file or a directory containing `model.json`.
- Current model constraints: mono, 16 kHz or 48 kHz, streaming-friendly 10 ms
  frames when `onnx_io.frame_samples` is declared.

Open Video:

- Runtime family: Open Video model packs consumed by Open CUDA, Open Vulkan, and
  task-specific Open Video effects.
- Root: `${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_video/<subject>/<pack_dir>/`.
- Pack shape: `model.json`, model artifacts declared by the manifest, and
  `LICENSE.txt`.
- Current tasks: `matting`, `face_detection`, `face_landmarks`,
  `video_denoise`, and `eye_contact`.
- Selection fields: video effects store `model_id` overrides for virtual
  background matting, auto frame face detection, eye contact, and video noise
  removal. Empty means "use the daemon-selected default".
- Production video selection is pack-based. The runtime should not grow ad hoc
  absolute model-path fields unless the canonical effect schema, diagnostics,
  and GUI behavior are updated deliberately.

The detailed user guides remain:

- `docs/open_source_audio_models_install.md`
- `docs/open_source_video_models_install.md`
- `docs/how_to_add_your_own_model.md`
- `docs/open_source_video_model_conversion_to_onnx.md`
- `docs/open_video_shared_inference.md`

## Core Rule

StudioCast supports bring-your-own models by validating and loading local model
packs through stable schemas. It should not silently infer arbitrary model
semantics from a filename, a repository URL, or an ONNX graph.

Every production model integration needs:

- A stable model ID.
- A declared task or supported effect.
- Local artifact paths that stay inside the pack directory.
- Tensor names, shapes, preprocessing, postprocessing, and runtime assumptions
  declared enough for the owning backend.
- License and provenance notes.
- Diagnostics that explain whether the model is installed, selected, active,
  missing, invalid, or blocked by runtime capability.

## Pack Ownership

Installed model packs are user data. Source-tree templates and curated assets
are project-maintained metadata.

- User-installed packs live under the XDG data model roots and may be added,
  removed, or replaced without rebuilding StudioCast.
- `resources/model_packs/**` should contain pack templates, license summaries,
  metadata, and only intentionally curated artifacts.
- Any committed model artifact is an exception that requires clear provenance,
  license compatibility, checksum metadata where practical, and a size/runtime
  reason.
- Placeholder packs must be explicit. A model ID containing `placeholder` is
  treated as not installed by current registries.
- Install scripts may download curated artifacts, but the live daemon and
  pipeline must not download models automatically.

## Manifest Rules

Model manifests are contracts, not comments.

Open Audio manifests must use the current `open_audio` schema:

- Required: `id`, `display_name`, `onnx_filename`.
- Optional but recommended: `origin.sha256`, `effects`, `sample_rate`,
  `channels`, and `onnx_io`.
- `onnx_filename` must be a safe relative path inside the pack.
- `channels` must currently be mono.
- `sample_rate` must currently be `16000` or `48000`.
- Declared `onnx_io.frame_samples` must match `sample_rate / 100`.
- Declared state inputs and state outputs must match in count.
- Auxiliary strength inputs must be scalar-shaped until the runtime supports
  richer control tensors.

Open Video manifests must use the current `open_video` schema:

- Required: `id`, `display_name`, `task`, and either legacy `onnx_filename` or
  schema v2 `files[]`.
- `files[].name` and legacy `onnx_filename` must be safe relative paths inside
  the pack.
- `task = "matting"` must include v1-style `input`, `output`, and `preprocess`
  metadata and at least one ONNX artifact.
- Schema v2 packs should declare each artifact with `kind`, optional `role`, and
  optional `sha256`.
- Dependencies such as face landmarks for eye contact should be represented in
  metadata rather than hidden in runtime code.

Do not add backend-specific manifest fields without documenting which runtime
owns them and what happens when another backend sees them.

## Selection Semantics

Model selection must be explicit, deterministic, and recoverable.

- Empty `model_id` means daemon-selected default, not "whatever was last used".
- Defaults must be deterministic and should favor a practical latency/quality
  tradeoff.
- A configured missing `model_id` must remain visible as missing until the user
  changes it or reinstalls the model.
- The GUI may display model lists and install hints, but the daemon decides which
  model is usable and active.
- Audio `model_path` wins over `model_id` because it is an explicit advanced
  override.
- Video production paths should select by `model_id` from the registry so
  status, persistence, and missing-model behavior stay coherent.
- Duplicate model IDs must be rejected or reported as problems; they must not
  produce order-dependent selection.

## Validation And Integrity

Validation should happen at install time, startup, model selection, explicit
self-test, or session creation. It must not become per-frame or per-audio-block
work.

Required default validation:

- Manifest is valid JSON.
- Required fields are present and type-correct.
- Referenced artifacts exist and are regular files.
- Manifest artifact paths are safe relative paths.
- The pack's task matches the consuming effect.
- Tensor metadata required by the backend is present.

Integrity verification:

- Install scripts should verify declared download checksums.
- `studiocast-open` self-tests and explicit verification paths may hash model
  files.
- Status polling may report cached or lightweight problems, but should not hash
  large model files repeatedly.
- Missing checksums may be reported as `unchecked`; that should not make the pack
  unusable unless the pack is curated and policy requires a checksum.

## Runtime Boundaries

StudioCast supports model contracts, not arbitrary model behavior.

- Open Audio models must fit the real-time audio framing contract or provide
  enough metadata for the processor to adapt safely.
- Open CUDA matting is GPU-only; missing CUDA, ONNX Runtime CUDA EP, or model
  packs should make the effect unavailable instead of silently running a CPU
  replacement.
- CPU fallback models should remain CPU-only, and GPU-backed models should avoid
  CPU/GPU bouncing as described in `docs/PIPELINE_POLICY.md`.
- Runtime sessions, model registries, tensor bindings, and buffers should be
  built once and reused until configuration, model ID/path, frame shape, provider,
  or runtime options change.
- Shared video analysis such as matte, face detections, and landmarks should be
  cached per capture sequence when multiple effects need the same result.
- A model failure in an optional effect should degrade that effect without taking
  down unrelated capture, playback, or pass-through paths when possible.

## Licensing And Provenance

Every model pack must make third-party terms visible.

- `LICENSE.txt` is required for curated packs and strongly required for user
  packs shown in GUI/install tooling.
- License notes should include upstream source, license type, and notable
  restrictions such as non-commercial use, attribution, or redistribution limits.
- StudioCast should not claim that a user-supplied model is redistributable,
  commercially usable, or safe for a particular deployment.
- Curated download scripts must preserve upstream license files or a clear
  license summary in the installed pack.
- New curated models with restrictive terms need an explicit product decision
  before they are presented as recommended defaults.

## Security And Privacy

Bring-your-own models are local code-adjacent assets and should be treated as
untrusted input.

- Do not execute scripts from a model pack as part of discovery or runtime
  loading.
- Do not follow manifest paths outside the pack directory.
- Do not fetch network resources from the daemon, GUI polling, or live pipeline
  based on model metadata.
- Keep model install actions user-initiated and visible.
- Prefer ONNX Runtime and existing native loaders over bespoke parsers for model
  formats.
- Error messages should expose local paths only in diagnostics/support contexts
  where the user asked for actionable setup information.

## GUI And Status Requirements

The daemon is authoritative for model availability and active runtime state.

Status should expose enough information for the GUI and CLI to answer:

- Which backend is active.
- Which model ID/path was requested.
- Which model ID/path was selected.
- Which installed models are available for the relevant task/effect.
- Which configured models are missing or invalid.
- Why a backend or effect is unavailable.
- Which install or self-test command is the next reasonable action.

The GUI must:

- Preserve model fields it does not expose.
- Show missing configured models instead of silently replacing them with
  defaults.
- Reconcile committed selections with daemon status.
- Avoid claiming a model is usable unless daemon diagnostics report it.
- Keep long-running installs, self-tests, and diagnostics off the UI thread.

## Documentation Requirements

Every new supported model family needs documentation before it is treated as
user-supported.

At minimum document:

- Install root and directory layout.
- Manifest schema and required fields.
- Supported tasks/effects.
- Runtime constraints such as sample rate, channels, frame size, image layout,
  tensor names, provider requirements, and GPU/CPU limitations.
- How to list installed packs.
- How to self-test a selected model.
- How to benchmark or smoke-test latency-sensitive paths.
- Known license restrictions and upstream sources.
- Fallback or degradation behavior when the model is missing or invalid.

If the implementation only supports curated models, say that directly. If users
can bring arbitrary compatible models, document the compatibility contract and
the expected failure modes.

## Review Checklist

Before accepting a model-related change, answer these questions:

- Does the change use an existing model registry, schema, and canonical effect
  field where one exists?
- Are model artifact paths constrained to the pack directory?
- Is selection deterministic when `model_id` is empty?
- Are missing configured models preserved visibly?
- Does status explain installed, missing, invalid, default, requested, and active
  model state?
- Are checksums and expensive validation kept out of the live path?
- Does the runtime rebuild sessions only on configuration or model changes?
- Are model license and provenance notes present?
- Does the GUI treat daemon status as authoritative?
- Is there a list/self-test/benchmark path for users and support?
- Are latency and CPU/GPU transfer implications consistent with
  `docs/LATENCY_POLICY.md` and `docs/PIPELINE_POLICY.md`?

