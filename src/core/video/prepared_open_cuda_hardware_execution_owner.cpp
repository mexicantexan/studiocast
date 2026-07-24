#include "core/video/prepared_open_cuda_hardware_execution_owner.h"

#include "core/video/effects/broadcast_effect_contract.h"

#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
#include "core/open_video/open_cuda_production_hal_owner.h"
#endif

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <utility>

namespace studiocast::video {
namespace {

constexpr std::uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kHashPrime = UINT64_C(1099511628211);
constexpr std::uint64_t kMaxPersistentBytes = UINT64_C(2) << 30;
constexpr std::uint64_t kMaxScratchBytes = UINT64_C(1) << 30;
constexpr std::uint64_t kMaxBoundaryBytes = UINT64_C(1) << 30;
constexpr std::uint32_t kMaxResources = 256;

std::uint64_t NextAdapterInstanceId() noexcept {
  static std::atomic<std::uint64_t> next{UINT64_C(0x4f430000)};
  std::uint64_t result = next.fetch_add(1, std::memory_order_relaxed);
  if (result == 0)
    result = next.fetch_add(1, std::memory_order_relaxed);
  return result;
}

std::uint64_t HashValue(std::uint64_t hash, std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    hash ^= static_cast<std::uint8_t>(value >> shift);
    hash *= kHashPrime;
  }
  return hash;
}

template <std::size_t Size>
std::uint64_t HashBytes(std::uint64_t hash,
                        const std::array<std::uint8_t, Size> &bytes) noexcept {
  for (const auto value : bytes) {
    hash ^= value;
    hash *= kHashPrime;
  }
  return hash;
}

std::uint64_t HashDomain(const hal::ExecutionDomainId &domain) noexcept {
  std::uint64_t hash = kHashOffset;
  hash = HashValue(hash, static_cast<std::uint64_t>(domain.backend));
  hash = HashValue(hash, static_cast<std::uint64_t>(domain.api_family));
  hash = HashValue(hash, domain.physical_device.vendor_id);
  hash = HashValue(hash, domain.physical_device.device_id);
  hash = HashBytes(hash, domain.physical_device.stable_id);
  hash = HashValue(hash, domain.logical_context_generation);
  hash = HashValue(hash, static_cast<std::uint64_t>(domain.allocator_family));
  hash =
      HashValue(hash, static_cast<std::uint64_t>(domain.memory_handle_family));
  hash = HashValue(hash, domain.queue_or_stream_identity);
  hash = HashValue(hash, static_cast<std::uint64_t>(domain.ownership_model));
  hash = HashValue(hash,
                   static_cast<std::uint64_t>(domain.synchronization_protocol));
  hash = HashValue(hash, domain.adapter_instance_id);
  return hash == 0 ? 1 : hash;
}

std::uint64_t HashArtifact(
    const open_video::ValidatedModelArtifactIdentity &artifact) noexcept {
  std::uint64_t hash = kHashOffset;
  hash = HashValue(hash, artifact.schema_version);
  hash = HashValue(hash, artifact.registry_generation);
  hash = HashBytes(hash, artifact.model_id_sha256);
  hash = HashBytes(hash, artifact.manifest_sha256);
  hash = HashBytes(hash, artifact.artifact_sha256);
  hash = HashBytes(hash, artifact.artifact_set_sha256);
  return hash == 0 ? 1 : hash;
}

int CanonicalStrength(int strength) noexcept {
  return std::clamp(strength, effects::contract::kVbStrengthMin,
                    effects::contract::kVbStrengthMax);
}

std::uint64_t
HashCanonicalRequest(const PreparedOpenCudaHardwareRequest &request) noexcept {
  std::uint64_t hash = kHashOffset;
  hash = HashValue(hash, request.stage_id);
  hash = HashValue(hash, static_cast<std::uint64_t>(request.semantic));
  hash = HashValue(hash, request.required);
  hash = HashValue(hash, request.capture_width);
  hash = HashValue(hash, request.capture_height);
  hash = HashValue(hash, request.output_width);
  hash = HashValue(hash, request.output_height);
  hash = HashValue(hash, request.host_input_is_tight_rgb24);
  hash = HashValue(hash, request.host_output_is_tight_rgb24);
  hash = HashValue(
      hash, static_cast<std::uint64_t>(CanonicalStrength(request.strength)));
  for (const auto component : request.remove_background_rgb)
    hash = HashValue(hash, component);
  hash =
      HashValue(hash, static_cast<std::uint64_t>(request.requested_provider));
  hash = HashValue(hash, request.wait.timeout_nanoseconds);
  hash = HashValue(hash, request.wait.poll_interval_nanoseconds);
  hash = HashValue(hash, request.wait.maximum_poll_count);
  hash = HashValue(hash, request.declares_writer_copy_cpu_tail);
  hash = HashValue(hash, HashArtifact(request.model_artifact));
  return hash == 0 ? 1 : hash;
}

bool Multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t *result) noexcept {
  if (!result ||
      (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool RequiredFrameBytes(const PreparedOpenCudaHardwareRequest &request,
                        std::uint64_t *bytes) noexcept {
  return Multiply(request.capture_width, request.capture_height, bytes) &&
         Multiply(*bytes, 3, bytes);
}

bool ValidBudget(const PreparedOpenCudaHardwareRequest &request) noexcept {
  const auto &budget = request.capacity;
  std::uint64_t frame_bytes = 0;
  return RequiredFrameBytes(request, &frame_bytes) && frame_bytes != 0 &&
         budget.persistent_bytes != 0 && budget.scratch_bytes != 0 &&
         budget.upload_staging_bytes >= frame_bytes &&
         budget.readback_staging_bytes >= frame_bytes &&
         budget.persistent_bytes <= kMaxPersistentBytes &&
         budget.scratch_bytes <= kMaxScratchBytes &&
         budget.upload_staging_bytes <= kMaxBoundaryBytes &&
         budget.readback_staging_bytes <= kMaxBoundaryBytes &&
         budget.resource_count != 0 && budget.resource_count <= kMaxResources &&
         budget.in_flight_slots == 1;
}

bool ValidWaitPolicy(const PreparedOpenCudaWaitPolicy &wait) noexcept {
  return wait.timeout_nanoseconds != 0 && wait.poll_interval_nanoseconds != 0 &&
         wait.poll_interval_nanoseconds <= wait.timeout_nanoseconds &&
         wait.maximum_poll_count != 0 && wait.stop_requested != nullptr &&
         wait.now_nanoseconds != nullptr && wait.poll_wait != nullptr;
}

bool CompatibleModelPack(
    const PreparedOpenCudaHardwareRequest &request) noexcept {
  const auto &pack = request.model_pack;
  if (!request.model_artifact.Valid() || pack.registry_generation == 0 ||
      pack.registry_generation != request.model_artifact.registry_generation ||
      pack.registry_manifest_sha256 != request.model_artifact.manifest_sha256 ||
      pack.task != "matting" || !pack.matting.has_value() ||
      pack.files.empty()) {
    return false;
  }
  const auto &matting = *pack.matting;
  return matting.input.width > 0 && matting.input.height > 0 &&
         matting.input.channels == 3 && matting.input.dtype == "float32" &&
         matting.input.layout == "nchw" && matting.output.kind == "alpha" &&
         matting.output.dtype == "float32" &&
         matting.preprocess.color == "rgb" &&
         matting.preprocess.range == "0..1";
}

hal::ImageDescriptor Image(std::uint32_t width, std::uint32_t height,
                           hal::HostCoherence coherence,
                           hal::Mutability mutability) noexcept {
  hal::ImageDescriptor image;
  image.width = width;
  image.height = height;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.alpha_meaning = hal::AlphaMeaning::none;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = coherence;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.in_place_permitted = false;
  image.base_alignment_bytes = 1;
  const std::uint64_t stride = static_cast<std::uint64_t>(width) * 3u;
  image.planes[0].row_stride_bytes = stride;
  image.planes[0].size_bytes = stride * height;
  image.planes[0].width = width;
  image.planes[0].height = height;
  image.planes[0].alignment_bytes = 1;
  return image;
}

hal::PreparedProgram Program(const PreparedOpenCudaHardwareRequest &request,
                             const hal::ExecutionDomainId &domain,
                             std::uint64_t program_id) noexcept {
  hal::PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = domain.adapter_instance_id;
  program.execution_domain = domain;
  program.host_input =
      Image(request.capture_width, request.capture_height,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  program.host_output =
      Image(request.output_width, request.output_height,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  // Setup-time owners supply these five authoritative generations. Exact
  // geometry, provider, artifact, and physical-device facts are additionally
  // bound by the three fixed invalidation hashes below; rewriting the typed
  // generations would make them disagree with executor setup evidence.
  program.invalidation_key.generations = request.generations;
  program.invalidation_key.generations.context =
      domain.logical_context_generation;
  program.invalidation_key.canonical_plan_hash = HashCanonicalRequest(request);
  program.invalidation_key.model_artifact_hash =
      HashArtifact(request.model_artifact);
  program.invalidation_key.execution_domain_hash = HashDomain(domain);
  program.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  program.budget = request.capacity;
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = true;
  program.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  program.strict_provider_selection = true;
  program.requested_execution_domain = domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;
  program.temporal.temporal = false;
  program.temporal.frame_drop_behavior =
      hal::FrameDropBehavior::latest_frame_wins;

  auto &stage = program.stages[0];
  stage.stage_id = request.stage_id;
  stage.semantic = request.semantic;
  stage.provider = hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  stage.backend = hal::BackendId::cuda;
  stage.execution_domain = domain;
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image =
      Image(request.capture_width, request.capture_height,
            hal::HostCoherence::not_host_visible, hal::Mutability::read_only);
  stage.output.kind = hal::StageIoKind::image;
  stage.output.image =
      Image(request.output_width, request.output_height,
            hal::HostCoherence::not_host_visible, hal::Mutability::read_write);
  stage.model.model_id_hash = request.model_artifact.model_id_sha256;
  stage.model.artifact_set_hash = request.model_artifact.artifact_set_sha256;
  stage.model.artifact_generation = request.model_artifact.registry_generation;
  stage.required = request.required;
  stage.failure_action = request.required
                             ? hal::FailureAction::fail_closed
                             : hal::FailureAction::fail_open_preserve_input;
  stage.circuit_breaker.consecutive_failure_limit = 2;
  stage.circuit_breaker.cooldown_frames = 30;
  stage.circuit_breaker.action =
      request.required ? hal::CircuitBreakerAction::quarantine_program
                       : hal::CircuitBreakerAction::disable_stage;
  program.stage_count = 1;

  auto &tail = program.cpu_tails[0];
  tail.id = hal::CpuTailId::writer_copy;
  tail.after_stage_id = stage.stage_id;
  tail.required = true;
  tail.placement = hal::DeclaredCpuTail::Placement::after_stage;
  tail.boundary_index = 1;
  program.cpu_tail_count = 1;
  return program;
}

bool SameRequestSemantics(
    const PreparedOpenCudaHardwareRequest &left,
    const PreparedOpenCudaHardwareRequest &right) noexcept {
  return left.effects_engine == right.effects_engine &&
         left.compute == right.compute &&
         left.requested_backend == right.requested_backend &&
         left.strict_backend_selection == right.strict_backend_selection &&
         left.requested_provider == right.requested_provider &&
         left.strict_provider_selection == right.strict_provider_selection &&
         left.tensorrt_enabled == right.tensorrt_enabled &&
         left.stage_id == right.stage_id && left.semantic == right.semantic &&
         left.semantic_stage_count == right.semantic_stage_count &&
         left.required == right.required &&
         left.single_resident_island == right.single_resident_island &&
         left.contains_other_execution_domain ==
             right.contains_other_execution_domain &&
         left.contains_other_resident_stage ==
             right.contains_other_resident_stage &&
         left.request_virtual_background_replace ==
             right.request_virtual_background_replace &&
         left.request_vignette == right.request_vignette &&
         left.request_video_noise_removal ==
             right.request_video_noise_removal &&
         left.request_eye_contact == right.request_eye_contact &&
         left.request_virtual_key_light == right.request_virtual_key_light &&
         left.request_auto_frame == right.request_auto_frame &&
         left.request_mirror == right.request_mirror &&
         left.shared_matte_required == right.shared_matte_required &&
         left.temporal_history_required == right.temporal_history_required &&
         left.capture_width == right.capture_width &&
         left.capture_height == right.capture_height &&
         left.output_width == right.output_width &&
         left.output_height == right.output_height &&
         left.host_input_is_tight_rgb24 == right.host_input_is_tight_rgb24 &&
         left.host_output_is_tight_rgb24 == right.host_output_is_tight_rgb24 &&
         left.scaling_required == right.scaling_required &&
         left.output_format_conversion_required ==
             right.output_format_conversion_required &&
         left.generations == right.generations &&
         left.model_artifact == right.model_artifact &&
         left.cuda_device_id == right.cuda_device_id &&
         CanonicalStrength(left.strength) ==
             CanonicalStrength(right.strength) &&
         left.remove_background_rgb == right.remove_background_rgb &&
         left.wait.timeout_nanoseconds == right.wait.timeout_nanoseconds &&
         left.wait.poll_interval_nanoseconds ==
             right.wait.poll_interval_nanoseconds &&
         left.wait.maximum_poll_count == right.wait.maximum_poll_count &&
         left.wait.stop_requested == right.wait.stop_requested &&
         left.wait.cancel_requested == right.wait.cancel_requested &&
         left.wait.now_nanoseconds == right.wait.now_nanoseconds &&
         left.wait.poll_wait == right.wait.poll_wait &&
         left.wait.callback_context == right.wait.callback_context &&
         left.declares_writer_copy_cpu_tail ==
             right.declares_writer_copy_cpu_tail &&
         left.capacity.persistent_bytes == right.capacity.persistent_bytes &&
         left.capacity.scratch_bytes == right.capacity.scratch_bytes &&
         left.capacity.upload_staging_bytes ==
             right.capacity.upload_staging_bytes &&
         left.capacity.readback_staging_bytes ==
             right.capacity.readback_staging_bytes &&
         left.capacity.resource_count == right.capacity.resource_count &&
         left.capacity.in_flight_slots == right.capacity.in_flight_slots &&
         left.model_pack.registry_generation ==
             right.model_pack.registry_generation &&
         left.model_pack.registry_manifest_sha256 ==
             right.model_pack.registry_manifest_sha256 &&
         left.model_pack.id == right.model_pack.id;
}

bool QualifiedInjectedRegistration(
    const hal::AdapterRegistration &registration,
    const PreparedOpenCudaHardwareRequest &request) noexcept {
  if (registration.adapter_instance_id == 0 ||
      registration.backend != hal::BackendId::cuda ||
      registration.resident_adapter == nullptr ||
      registration.execution_domain.backend != hal::BackendId::cuda ||
      registration.execution_domain.adapter_instance_id !=
          registration.adapter_instance_id ||
      !hal::ValidateExecutionDomain(registration.execution_domain)) {
    return false;
  }
  for (std::size_t index = 0; index < registration.semantic_capability_count;
       ++index) {
    const auto &capability = registration.semantic_capabilities[index];
    if (capability.semantic == request.semantic &&
        capability.provider ==
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
        hal::IsSelectableSemanticCapability(capability)) {
      return true;
    }
  }
  return false;
}

bool ExactPreparedReadiness(const hal::PreparedProgram &program,
                            const hal::ReadinessSnapshot &readiness) noexcept {
  return (readiness.status == hal::StatusId::ready ||
          readiness.status == hal::StatusId::degraded) &&
         readiness.available &&
         readiness.requested_backend == hal::BackendId::cuda &&
         readiness.selected_backend == hal::BackendId::cuda &&
         readiness.provider ==
             hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
         readiness.execution_domain == program.execution_domain &&
         readiness.ready_stage_mask == 1 && readiness.blocked_stage_mask == 0 &&
         readiness.disabled_stage_mask == 0;
}

} // namespace

PreparedOpenCudaHardwareEligibility EvaluatePreparedOpenCudaHardwareEligibility(
    const PreparedOpenCudaHardwareRequest &request) noexcept {
  if (request.effects_engine ==
      PreparedOpenCudaEffectsEngineIntent::automatic) {
    return PreparedOpenCudaHardwareEligibility::automatic_engine_compatibility;
  }
  if (request.effects_engine != PreparedOpenCudaEffectsEngineIntent::open_cuda)
    return PreparedOpenCudaHardwareEligibility::not_explicit_open_cuda;
  if (request.compute != PreparedOpenCudaComputeIntent::automatic &&
      request.compute != PreparedOpenCudaComputeIntent::cuda) {
    return PreparedOpenCudaHardwareEligibility::compute_family_disallowed;
  }
  if (request.requested_backend != hal::BackendId::cuda ||
      !request.strict_backend_selection) {
    return PreparedOpenCudaHardwareEligibility::not_strict_cuda;
  }
  if (request.requested_provider !=
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda ||
      !request.strict_provider_selection) {
    return PreparedOpenCudaHardwareEligibility::provider_mismatch;
  }
  if (request.tensorrt_enabled)
    return PreparedOpenCudaHardwareEligibility::tensorrt_unsupported;
  if (request.semantic_stage_count != 1 || request.stage_id == 0)
    return PreparedOpenCudaHardwareEligibility::stage_count_invalid;
  if (request.semantic != hal::SemanticEffectId::virtual_background_blur &&
      request.semantic != hal::SemanticEffectId::virtual_background_remove) {
    return PreparedOpenCudaHardwareEligibility::semantic_unsupported;
  }
  if (!request.single_resident_island ||
      request.contains_other_resident_stage) {
    return PreparedOpenCudaHardwareEligibility::mixed_resident_section;
  }
  if (request.contains_other_execution_domain)
    return PreparedOpenCudaHardwareEligibility::cross_domain_section;
  if (request.request_virtual_background_replace || request.request_vignette ||
      request.request_video_noise_removal || request.request_eye_contact ||
      request.request_virtual_key_light || request.request_auto_frame ||
      request.request_mirror) {
    return PreparedOpenCudaHardwareEligibility::unsupported_effect;
  }
  if (request.shared_matte_required)
    return PreparedOpenCudaHardwareEligibility::shared_analysis_unsupported;
  if (request.temporal_history_required)
    return PreparedOpenCudaHardwareEligibility::temporal_stage_unsupported;
  if (!request.host_input_is_tight_rgb24 ||
      !request.host_output_is_tight_rgb24 || request.capture_width == 0 ||
      request.capture_height == 0 || request.output_width == 0 ||
      request.output_height == 0) {
    return PreparedOpenCudaHardwareEligibility::unsupported_host_layout;
  }
  if (request.capture_width != request.output_width ||
      request.capture_height != request.output_height) {
    return PreparedOpenCudaHardwareEligibility::geometry_mismatch;
  }
  if (request.scaling_required)
    return PreparedOpenCudaHardwareEligibility::scaling_unsupported;
  if (request.output_format_conversion_required)
    return PreparedOpenCudaHardwareEligibility::output_conversion_unsupported;
  if (request.generations.configuration == 0 ||
      request.generations.geometry == 0 || request.generations.model == 0 ||
      request.generations.provider == 0 || request.generations.device == 0) {
    return PreparedOpenCudaHardwareEligibility::invalid_generation;
  }
  if (!request.model_artifact.Valid())
    return PreparedOpenCudaHardwareEligibility::
        model_artifact_evidence_incomplete;
  if (!CompatibleModelPack(request))
    return PreparedOpenCudaHardwareEligibility::model_pack_incompatible;
  if (!ValidBudget(request))
    return PreparedOpenCudaHardwareEligibility::resource_budget_invalid;
  if (!ValidWaitPolicy(request.wait))
    return PreparedOpenCudaHardwareEligibility::wait_policy_invalid;
  if (!request.declares_writer_copy_cpu_tail)
    return PreparedOpenCudaHardwareEligibility::writer_tail_invalid;
  return PreparedOpenCudaHardwareEligibility::eligible;
}

struct PreparedOpenCudaHardwareExecutionOwner::Impl {
  bool injected = false;
  hal::AdapterRegistration injected_registration{};
  std::unique_ptr<hal::PreparedProgramSession> injected_session;

#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
  // Session is declared after its adapter owner and is explicitly reset first.
  // The production owner therefore always outlives prepared session resources.
  std::unique_ptr<open_cuda::OpenCudaProductionHalOwner> production;
  std::unique_ptr<hal::PreparedProgramSession> production_session;
  std::unique_ptr<open_cuda::OpenCudaProductionHalOwner> pending_cleanup;
#endif

  PreparedOpenCudaHardwareRequest active_request{};
  PreparedOpenCudaHardwareExecutionSnapshot snapshot{};
  std::uint64_t next_program_id = 1;
  const std::uint64_t stable_adapter_instance_id = NextAdapterInstanceId();
  bool frame_pending = false;
  bool shutdown_requested = false;
  bool device_lost_latched = false;

  hal::PreparedProgramSession *Session() noexcept {
    if (injected)
      return injected_session.get();
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
    return production_session.get();
#else
    return nullptr;
#endif
  }

  const hal::PreparedProgramSession *Session() const noexcept {
    if (injected)
      return injected_session.get();
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
    return production_session.get();
#else
    return nullptr;
#endif
  }

#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
  static PreparedOpenCudaProductionLifecycleSnapshot
  Lifecycle(const open_cuda::OpenCudaProductionOwnerSnapshot &owner) noexcept {
    PreparedOpenCudaProductionLifecycleSnapshot result;
    result.status =
        owner.shutdown.shutdown_requested
            ? owner.shutdown.hal_status
            : (owner.bootstrap_domain_ready ? hal::StatusId::ready
                                            : hal::StatusId::blocked);
    result.reason = owner.shutdown.shutdown_requested
                        ? owner.shutdown.reason
                        : (owner.device_lost ? hal::ReasonId::device_lost
                                             : hal::ReasonId::none);
    result.shutdown_attempts = owner.shutdown.shutdown_attempts;
    result.bootstrap_domain_ready = owner.bootstrap_domain_ready;
    result.prepared_capability_committed = owner.prepared_capability.committed;
    result.quarantined = owner.quarantined;
    result.device_lost = owner.device_lost;
    result.shutdown_requested = owner.shutdown.shutdown_requested;
    result.shutdown_pending =
        owner.shutdown.shutdown_requested && !owner.shutdown.quiescent;
    result.quiescent = owner.shutdown.quiescent;
    result.driver_context_release_permitted =
        owner.shutdown.driver_context_release_permitted;
    result.stopped = owner.stopped;
    return result;
  }

  bool RetryPendingCleanup() noexcept {
    if (!pending_cleanup)
      return true;
    const auto shutdown = pending_cleanup->RequestShutdown();
    if (!shutdown.driver_context_release_permitted)
      return false;
    pending_cleanup.reset();
    return true;
  }

  void KeepOrRelease(
      std::unique_ptr<open_cuda::OpenCudaProductionHalOwner> owner) noexcept {
    if (!owner)
      return;
    const auto shutdown = owner->RequestShutdown();
    if (!shutdown.driver_context_release_permitted && !pending_cleanup)
      pending_cleanup = std::move(owner);
  }
#endif

  bool IncomingMatchesActive(
      const PreparedOpenCudaHardwareRequest &request) const noexcept {
    const auto *session = Session();
    if (!session || !session->prepared() ||
        !SameRequestSemantics(request, active_request)) {
      return false;
    }
    const auto projected = Program(request, session->program().execution_domain,
                                   session->program().program_id);
    return projected.invalidation_key == session->program().invalidation_key;
  }

  void Refresh(PreparedOpenCudaHardwareRoute route,
               PreparedOpenCudaHardwareEligibility eligibility) noexcept {
    snapshot.route = route;
    snapshot.eligibility = eligibility;
    snapshot.quarantined = false;
    snapshot.stopped = false;
    const auto *session = Session();
    if (session) {
      snapshot.readiness = session->readiness();
      snapshot.telemetry = session->telemetry();
      snapshot.latest_committed_frame = session->latest_committed_frame();
      snapshot.prepared = session->prepared();
      snapshot.quarantined = session->quarantined();
      snapshot.stopped = session->stopped();
      if (snapshot.prepared) {
        snapshot.active_program = session->program();
        snapshot.execution_domain = session->program().execution_domain;
      } else {
        snapshot.active_program = {};
        snapshot.execution_domain = {};
      }
    } else {
      snapshot.prepared = false;
      snapshot.active_program = {};
      snapshot.execution_domain = {};
    }
    snapshot.quarantined = snapshot.quarantined || device_lost_latched;
    if (injected) {
      snapshot.production.status =
          snapshot.stopped ? hal::StatusId::stopped
                           : (snapshot.prepared ? hal::StatusId::ready
                                                : hal::StatusId::blocked);
      snapshot.production.reason = snapshot.quarantined
                                       ? hal::ReasonId::device_lost
                                       : hal::ReasonId::none;
      snapshot.production.bootstrap_domain_ready = !snapshot.stopped;
      snapshot.production.prepared_capability_committed = snapshot.prepared;
      snapshot.production.quarantined = snapshot.quarantined;
      snapshot.production.device_lost = device_lost_latched;
      snapshot.production.shutdown_requested = snapshot.stopped;
      snapshot.production.quiescent = snapshot.stopped;
      snapshot.production.driver_context_release_permitted = snapshot.stopped;
      snapshot.production.stopped = snapshot.stopped;
    }
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
    else if (production) {
      snapshot.production = Lifecycle(production->Snapshot());
    } else if (pending_cleanup) {
      snapshot.production = Lifecycle(pending_cleanup->Snapshot());
    } else if (shutdown_requested) {
      snapshot.production = {};
      snapshot.production.status = hal::StatusId::stopped;
      snapshot.production.reason = hal::ReasonId::shutdown_requested;
      snapshot.production.shutdown_requested = true;
      snapshot.production.quiescent = true;
      snapshot.production.driver_context_release_permitted = true;
      snapshot.production.stopped = true;
    } else {
      snapshot.production = {};
    }
    snapshot.production.quarantined =
        snapshot.production.quarantined || device_lost_latched;
    snapshot.production.device_lost =
        snapshot.production.device_lost || device_lost_latched;
    if (device_lost_latched &&
        snapshot.production.reason == hal::ReasonId::none) {
      snapshot.production.reason = hal::ReasonId::device_lost;
    }
    snapshot.shutdown_pending =
        snapshot.production.shutdown_pending || pending_cleanup != nullptr;
#else
    snapshot.shutdown_pending = false;
#endif
  }

  void RetireActive(hal::ReasonId reason) noexcept {
    frame_pending = false;
    if (injected) {
      if (injected_session)
        injected_session->Invalidate(reason);
      active_request = {};
      return;
    }
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
    if (production_session) {
      production_session->Invalidate(reason);
      production_session->Shutdown();
      production_session.reset();
    }
    KeepOrRelease(std::move(production));
#endif
    active_request = {};
  }

  ~Impl() {
    if (injected_session && !injected_session->stopped())
      injected_session->Shutdown();
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
    if (production_session && !production_session->stopped())
      production_session->Shutdown();
    production_session.reset();
    KeepOrRelease(std::move(production));
    if (pending_cleanup) {
      const auto cleanup = pending_cleanup->RequestShutdown();
      if (cleanup.driver_context_release_permitted)
        pending_cleanup.reset();
      else
        (void)pending_cleanup.release();
    }
#endif
  }
};

PreparedOpenCudaHardwareExecutionOwner::PreparedOpenCudaHardwareExecutionOwner()
    : impl_(std::make_unique<Impl>()) {}

PreparedOpenCudaHardwareExecutionOwner::PreparedOpenCudaHardwareExecutionOwner(
    const hal::AdapterRegistration &registration)
    : impl_(std::make_unique<Impl>()) {
  impl_->injected = true;
  impl_->injected_registration = registration;
  impl_->injected_session = std::make_unique<hal::PreparedProgramSession>();
}

PreparedOpenCudaHardwareExecutionOwner::
    ~PreparedOpenCudaHardwareExecutionOwner() = default;

PreparedOpenCudaHardwareSetupResult
PreparedOpenCudaHardwareExecutionOwner::Configure(
    const PreparedOpenCudaHardwareRequest &request) noexcept {
  PreparedOpenCudaHardwareSetupResult result;
  const auto eligibility = EvaluatePreparedOpenCudaHardwareEligibility(request);
  result.eligibility = eligibility;

  if (impl_->frame_pending) {
    result.route = impl_->snapshot.prepared
                       ? PreparedOpenCudaHardwareRoute::hal_live
                       : PreparedOpenCudaHardwareRoute::legacy_compatibility;
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::output_not_committed;
    result.retained_previous_live_program = impl_->snapshot.prepared;
    impl_->Refresh(result.route, eligibility);
    return result;
  }

#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
  if (!impl_->injected && !impl_->RetryPendingCleanup()) {
    result.status = hal::StatusId::degraded;
    result.reason = hal::ReasonId::synchronization_failed;
    result.retained_previous_live_program =
        impl_->Session() && impl_->Session()->prepared();
    result.route = result.retained_previous_live_program
                       ? PreparedOpenCudaHardwareRoute::hal_live
                       : PreparedOpenCudaHardwareRoute::legacy_compatibility;
    impl_->Refresh(result.route,
                   PreparedOpenCudaHardwareEligibility::runtime_unavailable);
    return result;
  }
#endif

  const bool may_retain_old = impl_->IncomingMatchesActive(request);
  if (eligibility != PreparedOpenCudaHardwareEligibility::eligible) {
    if (!may_retain_old)
      impl_->RetireActive(hal::ReasonId::reconfiguration_required);
    result.status = hal::StatusId::blocked;
    result.reason =
        eligibility == PreparedOpenCudaHardwareEligibility::
                           model_artifact_evidence_incomplete ||
                eligibility ==
                    PreparedOpenCudaHardwareEligibility::model_pack_incompatible
            ? hal::ReasonId::capability_evidence_incomplete
            : hal::ReasonId::semantic_effect_unsupported;
    result.retained_previous_live_program =
        may_retain_old && impl_->Session() && impl_->Session()->prepared();
    result.route = result.retained_previous_live_program
                       ? PreparedOpenCudaHardwareRoute::hal_live
                       : PreparedOpenCudaHardwareRoute::legacy_compatibility;
    impl_->Refresh(result.route, eligibility);
    return result;
  }

  try {
    if (impl_->injected) {
      if (!QualifiedInjectedRegistration(impl_->injected_registration,
                                         request)) {
        result.status = hal::StatusId::blocked;
        result.reason = hal::ReasonId::semantic_provider_unavailable;
      } else {
        auto program =
            Program(request, impl_->injected_registration.execution_domain,
                    impl_->next_program_id++);
        hal::ReadinessSnapshot readiness;
        hal::ReasonId reason = hal::ReasonId::none;
        result.status = impl_->injected_session->Configure(
            program, *impl_->injected_registration.resident_adapter, &readiness,
            &reason);
        result.reason = reason;
        result.execution_domain = program.execution_domain;
        if (result.status == hal::StatusId::ready ||
            result.status == hal::StatusId::degraded) {
          if (ExactPreparedReadiness(program, readiness)) {
            impl_->active_request = request;
            impl_->device_lost_latched = false;
            result.route = PreparedOpenCudaHardwareRoute::hal_live;
            impl_->Refresh(result.route, eligibility);
            return result;
          }
          result.status = hal::StatusId::blocked;
          result.reason = hal::ReasonId::semantic_provider_unavailable;
        }
      }
      const bool retained =
          may_retain_old && impl_->injected_session->prepared();
      if (!retained)
        impl_->RetireActive(hal::ReasonId::reconfiguration_required);
      result.retained_previous_live_program = retained;
      result.route = retained
                         ? PreparedOpenCudaHardwareRoute::hal_live
                         : PreparedOpenCudaHardwareRoute::legacy_compatibility;
      impl_->Refresh(result.route,
                     PreparedOpenCudaHardwareEligibility::runtime_unavailable);
      return result;
    }

#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
    open_cuda::OpenCudaProductionBackendRequest backend_request;
    backend_request.adapter_instance_id = impl_->stable_adapter_instance_id;
    backend_request.cuda_device_id = request.cuda_device_id;
    backend_request.model_pack = request.model_pack;
    backend_request.model_artifact = request.model_artifact;
    backend_request.configuration_generation =
        request.generations.configuration;
    backend_request.model_preprocess_generation = request.generations.model;
    backend_request.model_artifact_hash = HashArtifact(request.model_artifact);
    backend_request.strength = CanonicalStrength(request.strength);
    backend_request.remove_background_rgb = request.remove_background_rgb;
    backend_request.wait_policy.timeout_nanoseconds =
        request.wait.timeout_nanoseconds;
    backend_request.wait_policy.poll_interval_nanoseconds =
        request.wait.poll_interval_nanoseconds;
    backend_request.wait_policy.maximum_poll_count =
        request.wait.maximum_poll_count;
    backend_request.wait_policy.stop_requested = request.wait.stop_requested;
    backend_request.wait_policy.cancel_requested =
        request.wait.cancel_requested;
    backend_request.wait_policy.now_nanoseconds = request.wait.now_nanoseconds;
    backend_request.wait_policy.poll_wait = request.wait.poll_wait;
    backend_request.wait_policy.callback_context =
        request.wait.callback_context;

    open_cuda::OpenCudaProductionPreparationResult preparation;
    auto candidate = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
        backend_request, &preparation);
    if (!candidate || !candidate->bootstrap_domain_ready() ||
        candidate->adapter() == nullptr) {
      result.status = preparation.hal_status;
      result.reason = preparation.reason;
      if (candidate)
        impl_->KeepOrRelease(std::move(candidate));
    } else {
      result.execution_domain = candidate->execution_domain();
      auto program = Program(request, candidate->execution_domain(),
                             impl_->next_program_id++);
      auto candidate_session = std::make_unique<hal::PreparedProgramSession>();
      hal::ReadinessSnapshot readiness;
      hal::ReasonId reason = hal::ReasonId::none;
      result.status = candidate_session->Configure(
          program, *candidate->adapter(), &readiness, &reason);
      result.reason = reason;
      if (result.status == hal::StatusId::ready ||
          result.status == hal::StatusId::degraded) {
        open_cuda::OpenCudaProductionPreparationResult capability;
        if (ExactPreparedReadiness(program, readiness)) {
          capability = candidate->CommitPreparedCapability(program, readiness);
        } else {
          capability.status = open_cuda::OpenCudaProductionOwnerStatus::
              capability_evidence_incomplete;
          capability.hal_status = hal::StatusId::blocked;
          capability.reason = hal::ReasonId::semantic_provider_unavailable;
        }
        if (capability.prepared_capability_committed) {
          impl_->RetireActive(hal::ReasonId::reconfiguration_required);
          impl_->production = std::move(candidate);
          impl_->production_session = std::move(candidate_session);
          impl_->active_request = request;
          impl_->device_lost_latched = false;
          result.route = PreparedOpenCudaHardwareRoute::hal_live;
          result.status = capability.hal_status;
          result.reason = capability.reason;
          impl_->Refresh(result.route, eligibility);
          return result;
        }
        result.status = capability.hal_status;
        result.reason = capability.reason;
      }
      candidate_session->Shutdown();
      impl_->KeepOrRelease(std::move(candidate));
    }
#else
    result.status = hal::StatusId::unavailable;
    result.reason = hal::ReasonId::strict_backend_unavailable;
#endif

    const bool retained =
        may_retain_old && impl_->Session() && impl_->Session()->prepared();
    if (!retained)
      impl_->RetireActive(hal::ReasonId::reconfiguration_required);
    result.retained_previous_live_program = retained;
    result.route = retained
                       ? PreparedOpenCudaHardwareRoute::hal_live
                       : PreparedOpenCudaHardwareRoute::legacy_compatibility;
    impl_->Refresh(result.route,
                   PreparedOpenCudaHardwareEligibility::runtime_unavailable);
    return result;
  } catch (const std::bad_alloc &) {
    result.status = hal::StatusId::resource_exhausted;
    result.reason = hal::ReasonId::resource_exhausted;
  } catch (...) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::capability_evidence_incomplete;
  }

  const bool retained =
      may_retain_old && impl_->Session() && impl_->Session()->prepared();
  if (!retained)
    impl_->RetireActive(hal::ReasonId::reconfiguration_required);
  result.retained_previous_live_program = retained;
  result.route = retained ? PreparedOpenCudaHardwareRoute::hal_live
                          : PreparedOpenCudaHardwareRoute::legacy_compatibility;
  impl_->Refresh(result.route,
                 PreparedOpenCudaHardwareEligibility::runtime_unavailable);
  return result;
}

PreparedOpenCudaHardwareFrameResult
PreparedOpenCudaHardwareExecutionOwner::ExecuteHostFrame(
    const std::uint8_t *input, std::size_t input_size_bytes,
    std::size_t input_row_stride_bytes, std::uint8_t *output,
    std::size_t output_size_bytes, std::size_t output_row_stride_bytes,
    std::uint64_t capture_sequence) noexcept {
  PreparedOpenCudaHardwareFrameResult frame;
  auto *session = impl_->Session();
  if (!session || !session->prepared()) {
    frame.status = session && session->quarantined()
                       ? hal::StatusId::quarantined
                       : hal::StatusId::stale;
    frame.reason = session && session->quarantined()
                       ? hal::ReasonId::device_lost
                       : hal::ReasonId::stale_program;
    return frame;
  }
  if (impl_->shutdown_requested) {
    frame.status = hal::StatusId::stopped;
    frame.reason = hal::ReasonId::shutdown_requested;
    return frame;
  }
  const auto &program = session->program();
  if (!input || !output || capture_sequence == 0 ||
      input_size_bytes != program.host_input.planes[0].size_bytes ||
      output_size_bytes != program.host_output.planes[0].size_bytes ||
      input_row_stride_bytes != program.host_input.planes[0].row_stride_bytes ||
      output_row_stride_bytes !=
          program.host_output.planes[0].row_stride_bytes) {
    frame.status = hal::StatusId::blocked;
    frame.reason = hal::ReasonId::host_frame_binding_invalid;
    return frame;
  }

  hal::HostFrameBinding binding;
  binding.input_descriptor = program.host_input;
  binding.output_descriptor = program.host_output;
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input);
  binding.input_planes[0].span_bytes = input_size_bytes;
  binding.input_planes[0].row_stride_bytes = input_row_stride_bytes;
  binding.output_planes[0].write_data = reinterpret_cast<std::byte *>(output);
  binding.output_planes[0].span_bytes = output_size_bytes;
  binding.output_planes[0].row_stride_bytes = output_row_stride_bytes;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = capture_sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;

  hal::HostFrameTokens tokens;
  hal::ReasonId reason = hal::ReasonId::none;
  frame.status = session->BindHostFrame(binding, &tokens, &reason);
  frame.reason = reason;
  if (frame.status != hal::StatusId::ready)
    return frame;

  hal::ExecuteRequest request;
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = capture_sequence;
  request.input = tokens.input;
  const auto execution = session->Execute(request);
  frame.status = execution.status;
  frame.reason = execution.reason;
  frame.completion = execution.completion;
  frame.transfers = execution.transfers;
  if (execution.stage_count == 1)
    frame.stage = execution.stages[0];
  frame.completion_pending = execution.completion.complete;

#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
  if (impl_->production)
    impl_->production->ObserveRuntimeStatus(execution.status, execution.reason);
#endif
  if (execution.reason == hal::ReasonId::device_lost)
    impl_->device_lost_latched = true;
  if (!execution.output_valid || !frame.completion_pending) {
    impl_->Refresh(PreparedOpenCudaHardwareRoute::hal_live,
                   PreparedOpenCudaHardwareEligibility::eligible);
    return frame;
  }

  hal::HostFrameOutputResolution resolution;
  frame.status = session->ResolveHostFrameOutput(
      execution.completion, execution.output, &resolution, &reason);
  frame.reason = reason;
  if (frame.status != hal::StatusId::ready || !resolution.output_ready ||
      resolution.capture_sequence != capture_sequence ||
      resolution.plane_count != 1 ||
      resolution.planes[0].write_data !=
          reinterpret_cast<std::byte *>(output)) {
    hal::ReasonId commit_reason = hal::ReasonId::none;
    (void)session->CommitOutput({execution.completion, false}, &commit_reason);
    frame.completion_pending = false;
    impl_->frame_pending = false;
    impl_->Refresh(PreparedOpenCudaHardwareRoute::hal_live,
                   PreparedOpenCudaHardwareEligibility::eligible);
    return frame;
  }

  frame.status = execution.status;
  frame.reason = execution.reason;
  frame.output = output;
  frame.output_size_bytes = output_size_bytes;
  frame.output_row_stride_bytes = output_row_stride_bytes;
  frame.output_ready = true;
  impl_->frame_pending = true;
  impl_->Refresh(PreparedOpenCudaHardwareRoute::hal_live,
                 PreparedOpenCudaHardwareEligibility::eligible);
  return frame;
}

hal::StatusId PreparedOpenCudaHardwareExecutionOwner::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  auto *session = impl_->Session();
  if (!session) {
    if (reason)
      *reason = hal::ReasonId::stale_program;
    return hal::StatusId::stale;
  }
  hal::ReasonId commit_reason = hal::ReasonId::none;
  const auto status = session->CommitOutput(
      {completion, output_write_succeeded}, &commit_reason);
  if (reason)
    *reason = commit_reason;
  if (status == hal::StatusId::ready ||
      (status == hal::StatusId::failed &&
       commit_reason == hal::ReasonId::output_write_failed)) {
    impl_->frame_pending = false;
  }
  impl_->Refresh(PreparedOpenCudaHardwareRoute::hal_live,
                 PreparedOpenCudaHardwareEligibility::eligible);
  return status;
}

void PreparedOpenCudaHardwareExecutionOwner::Invalidate(
    hal::ReasonId reason) noexcept {
  if (reason == hal::ReasonId::device_lost)
    impl_->device_lost_latched = true;
  impl_->RetireActive(reason);
  impl_->Refresh(PreparedOpenCudaHardwareRoute::legacy_compatibility,
                 PreparedOpenCudaHardwareEligibility::runtime_unavailable);
}

void PreparedOpenCudaHardwareExecutionOwner::ObserveDeviceLoss() noexcept {
  impl_->device_lost_latched = true;
  auto *session = impl_->Session();
  if (session)
    session->Invalidate(hal::ReasonId::device_lost);
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
  if (impl_->production) {
    impl_->production->ObserveRuntimeStatus(hal::StatusId::quarantined,
                                            hal::ReasonId::device_lost);
  }
#endif
  impl_->frame_pending = false;
  impl_->Refresh(PreparedOpenCudaHardwareRoute::hal_live,
                 PreparedOpenCudaHardwareEligibility::eligible);
}

void PreparedOpenCudaHardwareExecutionOwner::Shutdown() noexcept {
  impl_->shutdown_requested = true;
  impl_->frame_pending = false;
  if (impl_->injected) {
    if (impl_->injected_session && !impl_->injected_session->stopped())
      impl_->injected_session->Shutdown();
    impl_->active_request = {};
    impl_->Refresh(PreparedOpenCudaHardwareRoute::legacy_compatibility,
                   PreparedOpenCudaHardwareEligibility::runtime_unavailable);
    return;
  }
#if defined(STUDIOCAST_ENABLE_OPEN_CUDA) && STUDIOCAST_ENABLE_OPEN_CUDA
  if (impl_->production_session && !impl_->production_session->stopped()) {
    impl_->production_session->Shutdown();
  }
  if (impl_->production) {
    const auto shutdown = impl_->production->RequestShutdown();
    if (shutdown.driver_context_release_permitted) {
      impl_->production_session.reset();
      impl_->production.reset();
    }
  }
  (void)impl_->RetryPendingCleanup();
#endif
  impl_->active_request = {};
  impl_->Refresh(PreparedOpenCudaHardwareRoute::legacy_compatibility,
                 PreparedOpenCudaHardwareEligibility::runtime_unavailable);
}

PreparedOpenCudaHardwareExecutionSnapshot
PreparedOpenCudaHardwareExecutionOwner::Snapshot() const noexcept {
  return impl_->snapshot;
}

const PreparedOpenCudaHardwareRequest &
PreparedOpenCudaHardwareExecutionOwner::active_request() const noexcept {
  return impl_->active_request;
}

} // namespace studiocast::video
