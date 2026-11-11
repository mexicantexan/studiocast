# Changelog

## v0.2.9

This release focuses on Open Vulkan groundwork, normalized video compute backend
selection, and clearer diagnostics for GPU-backed video effects.

### Highlights

- Added optional Open Vulkan runtime plumbing with runtime-loaded Vulkan device,
  image, tensor, utility-kernel, and resize-kernel support.
- Introduced a normalized video compute backend preference for `auto`, `cpu`,
  `cuda`, and `vulkan` across daemon config, IPC, pipeline selection, GUI status,
  and `studiocastctl status`.
- Extended daemon and GUI status with resolved/active compute backend,
  provider/device details, tensor I/O mode, CPU-tail stages, fallback state, and
  degraded reasons.
- Reworked video pipeline backend decisions so explicit Vulkan requests report
  Vulkan fallback/degraded status instead of silently running CUDA.
- Added hardware-first Vulkan adapter discovery plus persistent, daemon-owned
  adapter selection for Intel, AMD, NVIDIA, and explicit CPU Vulkan fallback.
- Added a fail-closed production ncnn Vulkan matting contract and runtime
  lifecycle seam without promoting the CPU-transfer spike to production.
- Expanded optional hardware validation with Vulkan runtime smoke jobs and
  CPU/CUDA/Vulkan resize benchmarking.

### Added

- New persisted daemon config key: `video.compute.backend`, defaulting to
  `auto`.
- New `SET_VIDEO_CONFIG` field: `compute_backend=auto|cpu|cuda|vulkan`.
- New `video.compute` daemon status object and corresponding GUI status snapshot
  fields.
- Open Vulkan runtime loader, device, image, tensor, shader, SPIR-V header, and
  utility-kernel infrastructure.
- Vulkan kernels for RGB crop/resize, model preprocessing, alpha resize, alpha
  blur, RGB blur, alpha compositing, solid background compositing, and virtual
  key light.
- Optional ncnn Vulkan matting spike tooling behind
  `STUDIOCAST_ENABLE_NCNN_SPIKE`.
- Separate `STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING` production build gate,
  requiring Open Vulkan and a Vulkan-enabled ncnn dependency.
- Schema-v2 ncnn Vulkan model metadata for offline param/bin artifacts,
  checksums, blob names, converter/version details, and precision.
- A fakeable Vulkan matting runtime lifecycle with one-time graph load,
  persistent allocation, warmup, device-identity/residency checks, CPU-layer
  rejection, and latched fatal failures.
- Persisted `video.vulkan.device` and `video.vulkan.allow_cpu` configuration,
  stable adapter identities, cached candidates, and explicit missing or
  ambiguous selection diagnostics.
- A hardware-independent frame execution planner with unique per-dispatch
  parameter/descriptor slots, compute dependency barriers, and a final
  completion/readback boundary.
- Optional `studiocast-resize-backend-bench` benchmark behind
  `STUDIOCAST_BUILD_BENCHMARKS=ON`.
- Developer scripts for generating and validating embedded Vulkan SPIR-V
  headers.
- Setup helper options for Vulkan runtime packages, Mesa Vulkan ICD packages,
  shader tools, and explicit ONNX Runtime CPU/GPU flavor selection.
- Opt-in GitHub Actions Vulkan runtime smoke jobs for NVIDIA, AMD, and Intel
  runners.

### Changed

- Shared compute-backend helpers now centralize CUDA/Vulkan/CPU eligibility,
  active backend naming, and active GPU frame tracking.
- `auto` continues to prefer usable CUDA on NVIDIA systems; explicit `vulkan`
  now preserves the requested backend in diagnostics even when runtime fallback
  is required.
- Camera pipeline processing now tracks reusable frame artifacts more explicitly
  across matting, background replacement, auto-frame, key light, denoise, and
  eye-contact paths.
- Replacement-background caching now has a dedicated policy layer for cache
  reuse and invalidation decisions.
- Open Video/Open CUDA labels and setup text were clarified in the GUI, README,
  and developer docs.
- Default no-GPU CI explicitly disables Vulkan/CUDA kernels and keeps benchmark
  targets opt-in, while the enabled-features CI path compiles Open CUDA, Open
  Vulkan, Open Audio, dlib, and benchmarks without requiring GPU hardware.
- The setup helper now installs the ONNX Runtime GPU flavor only when
  `nvidia-smi` works or `--onnxruntime-flavor gpu` is requested; otherwise it
  installs the CPU flavor.
- The installer exposes independent Open Vulkan build, loader/diagnostic,
  Mesa Intel/AMD ICD, and shader-tool choices across install, update, repair,
  and clean-install workflows.
- Vulkan adapter auto-selection now prefers discrete, integrated, then virtual
  hardware and excludes CPU implementations unless explicitly requested.
- Background blur and alpha composite share one blocking submission: the
  normal blur path is reduced from two submissions to one, and the feathered
  path from three submissions to two.

### Fixed

- Fixed audit findings around Vulkan virtual-background readiness, backend
  fallback reporting, degraded status, and CPU/pass-through behavior.
- Prevented explicit Vulkan requests from being represented as CUDA-backed
  execution when Vulkan is unavailable or not production-ready for an effect.
- Improved GUI engine status handling for unreachable or unparsable daemon
  status, including microphone and speaker active-backend labels.
- Tightened Open Vulkan setup policy behavior for disabled builds, missing
  runtime support, missing devices, and unavailable matting support.
- Improved replacement-background cache invalidation for source frames, model
  changes, target size changes, background color/path changes, and alpha reuse.
- Reduced misleading optional-GPU probe/tool output around runtime checks.

### Test Coverage

- Added or expanded tests for daemon status JSON, GUI status snapshots, video
  service compute-backend behavior, Vulkan setup policy, replacement-background
  cache policy, Vulkan kernel wrappers, frame artifact caching, installer
  backend setup behavior, CMake backend cache behavior, and CUDA driver API
  fallback handling.
- Added no-GPU coverage for production ncnn dependency policy, manifest
  completeness/checksums, runtime lifecycle and failure latching, persistent
  adapter selection, missing/ambiguous devices, CPU-device policy, and frame
  submission/barrier planning.

### Compatibility Notes

- Existing configs that omit `video.compute.backend` continue to resolve as
  `auto`.
- Open Vulkan remains optional and experimental. Build with
  `-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON` to compile it.
- Installing Vulkan loader or ICD packages does not guarantee usable hardware
  support; runtime status reports device and fallback details.
- Production Vulkan virtual-background matting remains unavailable until a
  real ncnn adapter shares StudioCast's Vulkan device/queue/allocation domain,
  runs a reviewed GPU-only graph, and proves device-resident input, alpha, and
  output. The existing ncnn spike still uses CPU `ncnn::Mat` input/output and
  is not a production fallback.
- Persisted Vulkan adapter identities use stable device properties rather than
  run-local enumeration indices. Indistinguishable identical adapters fail
  closed pending a stronger UUID/PCI identity.
- Open CUDA still requires an ONNX Runtime build with CUDA Execution Provider
  support and valid Open Video model packs. On machines where `nvidia-smi` is
  unavailable during setup, pass `--onnxruntime-flavor gpu` when CUDA support is
  required.
- The top-level `VERSION` file still reads `0.2.8` on this branch. Merging
  `v0.2.9` to `master` should trigger the normal version-bump flow before
  tagging the release.

## v0.2.7

This release focuses on live video performance, virtual camera format control,
and more resilient audio/video fallback behavior.

### Highlights

- Added virtual camera output format selection for `rgb24` and `yuyv` in the
  GUI, daemon config, daemon CLI, `studiocastctl video set`, and status output.
- Reworked hot RGB/YUYV/RGB-BGR conversion paths with runtime-selected scalar,
  optional `libyuv`, SSSE3, SSE4.1, and AVX2 backends.
- Improved CPU resize, RGB output prep, background-mask generation, and GPU
  scaler decisions to reduce unnecessary work on live frames.
- Made microphone source handling more explicit and robust, including safe
  auto-selection, unavailable-source reporting, and pass-through fallback when
  Maxine microphone effects fail.
- Expanded video, audio, daemon status, installer, and Maxine path test coverage
  around the new behavior.

### Added

- New persisted config key: `video.output_format`, defaulting to `rgb24`.
- New daemon option: `studiocastd --output-format rgb24|yuyv`.
- New CLI setting: `studiocastctl video set output_format=rgb24|yuyv`.
- GUI video setup control for `RGB24 / RGB3` and `YUYV 4:2:2` output.
- Status fields for requested output format, negotiated capture/output formats,
  capture fallback state, capture fallback reason, microphone source
  availability, source errors, and source warnings.
- Developer benchmark builds behind `STUDIOCAST_BUILD_BENCHMARKS=ON`, including
  RGB/YUYV/RGB-BGR conversion and CUDA transfer benchmark tools.

### Changed

- Virtual camera output now honors the requested FourCC instead of silently
  cycling through alternate output formats.
- V4L2 capture negotiation now treats 720p and larger YUYV requests as
  MJPEG-worthy when MJPEG preference is enabled.
- MJPEG capture decode failures can fall back once to raw YUYV and report that
  fallback through daemon/GUI status.
- Camera idle and preview status preserve configured input/output device names
  instead of hiding them while the heavy pipeline is stopped.
- Maxine SDK library discovery now gives explicit SDK roots priority over
  unrelated loader-path libraries, while still keeping system-loader fallback.
- `studiocastctl debug-report` now includes bounded PulseAudio snapshots for
  `pactl info`, defaults, sources, sinks, and loaded modules.

### Fixed

- Preserved configured but disconnected microphone sources and reported them as
  unavailable instead of silently changing user configuration.
- Avoided unsafe StudioCast/monitor sources during automatic microphone source
  selection.
- Fell back to pass-through audio with cooldown after Maxine microphone setup
  failures to avoid repeated restart churn.
- Skipped inactive standalone GPU scaler transfers when CPU resize is allowed
  and no same-backend GPU effect or deferred GPU output needs reuse.
- Reduced redundant per-frame work in CPU resize, padded RGB output prep, and
  CPU background removal.
- Tightened CUDA context validation for the retained primary context and avoids
  repeated validation after success.

### Performance

Local benchmark validation selected AVX2 for RGB/YUYV paths and SSSE3 for
RGB/BGR on the test machine. These numbers are local measurements, not fixed
runtime guarantees.

| Path | Local before -> after | Change |
|---|---:|---:|
| RGB24 -> YUYV selected dispatch | 11.047 ms -> 3.387 ms | 69.3% lower |
| YUYV -> RGB24 selected dispatch | 29.904 ms -> 11.286 ms | 62.3% lower |
| RGB24/BGR24 selected dispatch | 4.328 ms -> 1.351 ms | 68.8% lower |
| RGB24 resize planning, 1280x720 -> 1920x1080 | 129.998 ms -> 92.306 ms | 29.0% lower |
| RGB24 resize hot loop, 1280x720 -> 1920x1080 | 97.037 ms -> 45.092 ms | 53.5% lower |

### Compatibility Notes

- Existing configs that omit `video.output_format` continue to default to
  `rgb24`; invalid persisted values fall back to `rgb24`.
- Changing the virtual camera output format restarts the camera pipeline.
- Output negotiation is stricter about the requested format. Systems whose
  loopback device rejects `rgb24` may need to choose `yuyv` explicitly.
- Debug reports now include local PulseAudio device/module snapshots. Review
  reports before sharing them publicly.
- The top-level `VERSION` file still reads `0.2.6` on this branch; update it
  through the release/version-bump flow before tagging `v0.2.7`.

### Verification

- Focused CTest subset passed locally:
  `studiocast-audio-tests`, `studiocast-maxine-paths-tests`,
  `studiocast-daemon-status-tests`, `studiocast-installer-backend-tests`,
  `studiocast-video-tests`, and `studiocast-v4l2loopback-diagnostics-tests`.
- `./build/studiocast-probe --self-test` passed with `SELFTEST OK`.
- PR verification also included building `studiocast-video-tests`,
  `studiocastd`, `studiocastctl`, and `studiocast`, running
  `ctest --test-dir build -R '^studiocast-video-tests$' --output-on-failure`,
  and running `git diff --check`.
