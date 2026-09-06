# HAL parity completion and qualification

- Status: Active migration and release contract
- Applies To: Prepared HAL ownership, compatibility-route removal, and backend
  capability claims
- Audience: Video runtime, backend, daemon, test, and release maintainers
- Last Verified: 2026-09-06

The Hardware Execution Layer is StudioCast's intended video execution
architecture. Neutral contracts, planning, and ownership are necessary
milestones, but none of them alone proves parity with the existing production
pipeline. Parity means that the prepared route preserves the established
selection, effect, failure, output, status, and rollback behavior for a bounded
family on supported hardware.

## Migration order

Complete existing-functionality Open CUDA parity first. This includes strict
CUDA requests, `auto` selection and fallback, each already supported effect,
and supported combinations and dependencies such as Auto Frame. The current
prepared CUDA route does not yet cover every one of these cases. Uncovered
behavior must remain on a bounded compatibility route or remain explicitly
blocked; neutral ownership must not be used to claim that it is fixed. Any
regression of behavior that worked before the migration must be restored for
the CUDA parity milestone. A compatibility route is temporary preservation,
not an acceptable permanent parity result.

After CUDA parity, expand production Vulkan in bounded phases. Each phase must
name its effects, adapter and execution-domain contract, models and artifacts,
hardware matrix, transfer budget, failure behavior, and rollback boundary.
Passing one phase does not promote other Vulkan effects.

Vulkan matting remains blocked until all of the following exist and are
reviewed together:

- shared-device GPU-only inference using StudioCast's selected physical and
  logical device, queue, buffers, ownership, and synchronization protocol;
- converted schema-v2 ncnn Vulkan parameter/bin artifacts with pinned,
  verified checksums and the required tensor/converter metadata;
- no CPU layers, host inference input/output, intermediate readback, or hidden
  CPU tail; and
- real-hardware execution evidence for every device family claimed by the
  release.

## Family parity gate

Before removing a working route or describing a family as prepared-HAL
complete, record evidence that:

- canonical engine/compute preferences resolve identically, including strict
  requests, `auto`, fallback, unavailable, and degraded outcomes;
- every existing single effect and supported combination has equivalent stage
  order, shared dependencies, parameter behavior, and output semantics;
- model selection, artifact identity, missing/invalid-model behavior, and
  provider requirements remain deterministic and daemon-owned;
- optional failure, circuit breaking, device loss, reconfiguration, frame
  drop, stop, and retry behavior preserve valid output and bounded latency;
- writer-commit attribution, per-effect status, blocker reasons, fallback
  reasons, transfers, CPU tails, and active backend facts remain truthful;
- focused regression and integration tests cover the family, and applicable
  real-hardware tests pass on the supported matrix; and
- a reviewed rollback keeps or restores the previous working route without
  changing persisted canonical configuration.

Compatibility may remain only where it preserves existing behavior while a
known parity gap is open. Its effect coverage must be explicit in typed status,
it must obey strict-selection and GPU-only rules, and it must not be counted as
prepared-HAL capability or completion evidence.

## Route-removal gate

A compatibility route or legacy direct dependency may be eliminated only when
the family parity evidence above is reviewed and the replacement is enabled in
the same configurations and supported hardware contexts as the working route.
The change must include a tested rollback boundary. Source ownership or CMake
dependency cleanup is insufficient evidence by itself.

## Qualification and release gate

A production capability or release note may claim only the exact effects and
hardware covered by reviewed evidence. Synthetic adapters, loader/device
discovery, compilation, unit conformance, and a neutral manager cutover do not
substitute for model-backed real-hardware execution. Release qualification
must retain the existing policies in [PIPELINE_POLICY.md](PIPELINE_POLICY.md),
[LATENCY_POLICY.md](LATENCY_POLICY.md), [MODEL_POLICY.md](MODEL_POLICY.md), and
[MANUAL_TESTING.md](MANUAL_TESTING.md), plus backend-specific installer
evidence such as
[INSTALLER_VULKAN_CAPABILITIES.md](INSTALLER_VULKAN_CAPABILITIES.md).

Local family parity completion does not itself qualify a production capability
or release. Until every applicable release gate is satisfied, documentation
and PR text must name the remaining compatibility or blocked routes and avoid
claiming production readiness or release completion. Until the family parity
gate is satisfied, they must also avoid claiming end-to-end parity.
