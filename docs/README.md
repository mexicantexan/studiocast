# StudioCast Documentation Index

- Status: Active documentation lifecycle index
- Applies To: The current repository documentation
- Audience: Users, contributors, maintainers, and reviewers
- Last Verified: 2026-11-11

Repository policies are discovered with `docs/*POLICY*.md`; all policy files
therefore remain at the `docs/` root. Active cross-subsystem contracts also stay
at the root when source or test consistency checks depend on their stable path.
Historical audits live under `docs/archive/audits/` and are evidence about a
named baseline, not statements about the current implementation.

## Policies

These documents are authoritative review requirements.

| Document | Status | Applies To | Owner/Audience | Last Verified |
| --- | --- | --- | --- | --- |
| [GUI_POLICY.md](GUI_POLICY.md) | Active policy | GUI authority, IPC, and runtime presentation | GUI/runtime maintainers | 2026-07-22 |
| [LATENCY_POLICY.md](LATENCY_POLICY.md) | Active policy | Live audio/video latency and scheduling | Runtime/performance maintainers | 2026-07-22 |
| [MODEL_POLICY.md](MODEL_POLICY.md) | Active policy | Model packs, selection, integrity, and licensing | Model/installer maintainers | 2026-07-22 |
| [PIPELINE_POLICY.md](PIPELINE_POLICY.md) | Active policy | CPU/GPU transfer and execution behavior | Video/backend maintainers | 2026-07-22 |

## Active contracts

These files describe current architecture or stable cross-subsystem behavior.

| Document | Status | Applies To | Owner/Audience | Last Verified |
| --- | --- | --- | --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Active contract | Target layers, daemon authority, effects, and model layout | All contributors | 2026-07-23 |
| [HARDWARE_EXECUTION_LAYER.md](HARDWARE_EXECUTION_LAYER.md) | Active staged-migration contract | Neutral HAL, prepared adapters, bounded evidence and qualification gates, execution domains, and typed status | Video/backend maintainers | 2026-11-11 |
| [HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md) | Active migration and release contract | Family parity, route removal, and bounded hardware qualification | Video/release maintainers | 2026-11-11 |
| [INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md) | Active contract | Curated model catalog and transactions | Installer/model maintainers | 2026-11-11 |
| [INSTALLER_VULKAN_CAPABILITIES.md](INSTALLER_VULKAN_CAPABILITIES.md) | Active capability contract | Per-effect installer/runtime Vulkan evidence | Installer/Vulkan maintainers | 2026-07-22 |
| [RELEASE_CHANNEL.md](RELEASE_CHANNEL.md) | Active security contract | Stable release transport, signing, and updates | Release/security maintainers | 2026-07-22 |
| [SYSTEM_HELPER_PACKAGING.md](SYSTEM_HELPER_PACKAGING.md) | Active security contract | Privileged helper and polkit packaging | Packaging/security maintainers | 2026-07-22 |
| [V4L2_WRITER_TRANSPORT.md](V4L2_WRITER_TRANSPORT.md) | Active performance contract | V4L2 writer transport semantics | Video/performance maintainers | 2026-07-22 |

`INSTALLER_VULKAN_CAPABILITIES.md` is intentionally not archived. Source and
test consistency checks consume it as the active installer capability matrix.

## User and developer guides

| Document | Purpose | Audience | Status |
| --- | --- | --- | --- |
| [SETUP.md](SETUP.md) | Source setup, installer entry points, and optional backends | Users and developers | Maintained guide |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Repository, build, runtime, test, and release workflows | Contributors | Maintained guide |
| [MANUAL_TESTING.md](MANUAL_TESTING.md) | Hardware and desktop regression checklist | Testers and maintainers | Maintained guide |
| [maxine_install.md](maxine_install.md) | User-supplied Maxine SDK setup | Maxine users | Maintained guide |
| [open_source_audio_models_install.md](open_source_audio_models_install.md) | Open Audio model installation and schema | Open Audio users | Maintained guide |
| [open_source_video_models_install.md](open_source_video_models_install.md) | Open CUDA prerequisites and model installation | Open Video users | Maintained guide |
| [how_to_add_your_own_model.md](how_to_add_your_own_model.md) | Open Video pack taxonomy and manifest schema | Model authors | Maintained guide |
| [open_source_video_model_conversion_to_onnx.md](open_source_video_model_conversion_to_onnx.md) | Model sourcing and conversion notes | Model authors | Maintained guide |
| [TRADEMARKS.md](TRADEMARKS.md) | Trademark and affiliation notice | All audiences | Active legal note |

## Design and measurement notes

These documents inform implementation work but do not override policies or
active capability contracts.

| Document | Status | Baseline or Scope | Audience |
| --- | --- | --- | --- |
| [open_video_shared_inference.md](open_video_shared_inference.md) | Shared-analysis design and prepared-runtime checkpoint | Same-frame analysis, temporal/transfer boundaries, and latest-frame scheduling | Video/model maintainers |
| [vulkan_matting_ncnn_spike.md](vulkan_matting_ncnn_spike.md) | Experimental design record | Optional ncnn Vulkan spike | Vulkan/model maintainers |
| [V4L2_WRITER_MEASUREMENTS.md](V4L2_WRITER_MEASUREMENTS.md) | Historical measurement record | Commits named in the document | Performance maintainers |

## Historical audits

Archived audits retain dated evidence and design history. They are not current
implementation descriptions.

| Document | Audited Baseline | Audience | Archived |
| --- | --- | --- | --- |
| [Installer backend audit](archive/audits/INSTALLER_BACKEND_AUDIT.md) | `5f3bc485a4dbb2d3a463945591f910fbab53c9a6` | Installer maintainers | 2026-07-22 |
| [Installer GUI audit](archive/audits/INSTALLER_GUI_AUDIT.md) | v0.2.9 plus named Wave 1 proposals | Installer GUI maintainers | 2026-07-22 |
| [Installer privilege audit](archive/audits/INSTALLER_PRIVILEGE_AUDIT.md) | `5f3bc485a4dbb2d3a463945591f910fbab53c9a6` | Security/installer maintainers | 2026-07-22 |
| [GUI strengths audit](archive/audits/gui_strengths_audit.md) | GUI observed 2026-07-03 | GUI designers | 2026-07-22 |
| [GUI weaknesses audit](archive/audits/gui_weaknesses_audit.md) | GUI observed 2026-07-03 | GUI designers | 2026-07-22 |

The empty historical roadmap placeholder was removed. Current behavior and
planned work must be documented in an active contract, tracked issue, or an
explicitly baselined design note rather than an unmaintained placeholder.
