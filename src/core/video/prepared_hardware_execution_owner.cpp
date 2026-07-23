#include "core/video/prepared_hardware_execution_owner.h"

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/vulkan/hal_resident_adapter.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/qualified_final_resident_executor.h"
#endif

#include <atomic>
#include <limits>
#include <string>

namespace studiocast::video {
namespace {

constexpr std::uint32_t kVignetteStageId = 8;
constexpr std::uint32_t kMirrorStageId = 9;

bool Multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t *result) noexcept {
  if (!result ||
      (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool Add(std::uint64_t *value, std::uint64_t amount) noexcept {
  if (!value || amount > std::numeric_limits<std::uint64_t>::max() - *value) {
    return false;
  }
  *value += amount;
  return true;
}

std::uint64_t HashValue(std::uint64_t hash, std::uint64_t value) noexcept {
  constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
  for (unsigned shift = 0; shift < 64; shift += 8) {
    hash ^= (value >> shift) & UINT64_C(0xff);
    hash *= kPrime;
  }
  return hash;
}

std::uint64_t
HashRequest(const PreparedFinalHardwareRequest &request) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  hash = HashValue(hash, static_cast<std::uint64_t>(request.requested_backend));
  hash = HashValue(hash, request.request_fixed_center_vignette);
  hash = HashValue(hash, request.request_mirror);
  hash = HashValue(
      hash, static_cast<std::uint64_t>(request.vignette_intensity_percent));
  hash = HashValue(hash, request.input_width);
  hash = HashValue(hash, request.input_height);
  hash = HashValue(hash, request.output_width);
  hash = HashValue(hash, request.output_height);
  return hash == 0 ? 1 : hash;
}

std::uint64_t HashDomain(const hal::ExecutionDomainId &domain) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  hash = HashValue(hash, static_cast<std::uint64_t>(domain.backend));
  hash = HashValue(hash, static_cast<std::uint64_t>(domain.api_family));
  hash = HashValue(hash, domain.physical_device.vendor_id);
  hash = HashValue(hash, domain.physical_device.device_id);
  for (const auto byte : domain.physical_device.stable_id)
    hash = HashValue(hash, byte);
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

hal::CanonicalStageIntent Intent(std::uint32_t stage_id,
                                 hal::SemanticEffectId semantic) noexcept {
  hal::CanonicalStageIntent intent;
  intent.stage_id = stage_id;
  intent.semantic = semantic;
  intent.required = false;
  intent.failure_action = hal::FailureAction::fail_open_preserve_input;
  return intent;
}

bool SameLiveProgram(const PreparedFinalHardwareRequest &left,
                     const PreparedFinalHardwareRequest &right) noexcept {
  return left.input_width == right.input_width &&
         left.input_height == right.input_height &&
         left.output_width == right.output_width &&
         left.output_height == right.output_height &&
         left.requested_backend == right.requested_backend &&
         left.strict_backend_selection == right.strict_backend_selection &&
         left.request_fixed_center_vignette ==
             right.request_fixed_center_vignette &&
         left.request_mirror == right.request_mirror &&
         left.vignette_intensity_percent == right.vignette_intensity_percent &&
         left.host_input_is_tight_rgb24 == right.host_input_is_tight_rgb24 &&
         left.host_output_is_tight_rgb24 == right.host_output_is_tight_rgb24;
}

hal::PreparedProgram Program(const PreparedFinalHardwareRequest &request,
                             const hal::AdapterRegistration &registration,
                             std::uint64_t program_id) noexcept {
  hal::PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = registration.adapter_instance_id;
  program.execution_domain = registration.execution_domain;
  program.host_input =
      Image(request.input_width, request.input_height,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  program.host_output =
      Image(request.output_width, request.output_height,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  program.invalidation_key.generations = request.generations;
  if (program.invalidation_key.generations.provider == 0)
    program.invalidation_key.generations.provider = 1;
  if (program.invalidation_key.generations.device == 0)
    program.invalidation_key.generations.device = 1;
  program.invalidation_key.generations.context =
      registration.execution_domain.logical_context_generation;
  program.invalidation_key.canonical_plan_hash = HashRequest(request);
  program.invalidation_key.execution_domain_hash =
      HashDomain(registration.execution_domain);
  program.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  program.requested_backend = hal::BackendId::vulkan;
  program.strict_backend_selection = true;
  program.requested_provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  program.strict_provider_selection = true;
  program.requested_execution_domain = registration.execution_domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;

  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t mask_bytes = 0;
  if (!Multiply(request.input_width, request.input_height, &input_bytes) ||
      !Multiply(input_bytes, 3, &input_bytes) ||
      !Multiply(request.output_width, request.output_height, &output_bytes) ||
      !Multiply(output_bytes, 3, &output_bytes) ||
      !Multiply(request.output_width, request.output_height, &mask_bytes) ||
      !Multiply(mask_bytes, sizeof(float), &mask_bytes)) {
    return {};
  }
  program.budget.upload_staging_bytes = input_bytes;
  program.budget.readback_staging_bytes = output_bytes;
  program.budget.scratch_bytes = output_bytes;
  program.budget.persistent_bytes = input_bytes;
  program.budget.resource_count = 4;
  program.budget.in_flight_slots = 1;
  if (request.request_fixed_center_vignette) {
    if (!Add(&program.budget.persistent_bytes, output_bytes) ||
        !Add(&program.budget.persistent_bytes, mask_bytes)) {
      return {};
    }
    program.budget.resource_count += 2;
  }
  if (request.request_mirror) {
    if (!Add(&program.budget.persistent_bytes, output_bytes))
      return {};
    ++program.budget.resource_count;
  }

  const auto input =
      Image(request.input_width, request.input_height,
            hal::HostCoherence::not_host_visible, hal::Mutability::read_only);
  const auto output =
      Image(request.output_width, request.output_height,
            hal::HostCoherence::not_host_visible, hal::Mutability::read_write);
  const auto final_input =
      Image(request.output_width, request.output_height,
            hal::HostCoherence::not_host_visible, hal::Mutability::read_only);
  const auto add_stage = [&](std::uint32_t stage_id,
                             hal::SemanticEffectId semantic,
                             const hal::ImageDescriptor &stage_input) {
    auto &stage = program.stages[program.stage_count++];
    stage.stage_id = stage_id;
    stage.semantic = semantic;
    stage.provider =
        hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    stage.backend = hal::BackendId::vulkan;
    stage.execution_domain = program.execution_domain;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = stage_input;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = output;
    stage.required = false;
    stage.failure_action = hal::FailureAction::fail_open_preserve_input;
    stage.circuit_breaker.consecutive_failure_limit = 1;
    stage.circuit_breaker.cooldown_frames = 30;
    stage.circuit_breaker.action = hal::CircuitBreakerAction::disable_stage;
  };
  if (request.request_fixed_center_vignette)
    add_stage(kVignetteStageId, hal::SemanticEffectId::vignette, input);
  if (request.request_mirror) {
    add_stage(kMirrorStageId, hal::SemanticEffectId::mirror,
              request.request_fixed_center_vignette ? final_input : input);
  }
  return program;
}

hal::ResolutionRequest
Resolution(const PreparedFinalHardwareRequest &request) noexcept {
  hal::ResolutionRequest resolution;
  resolution.backend.mode = hal::BackendRequestMode::explicit_backend;
  resolution.backend.backend = hal::BackendId::vulkan;
  resolution.backend.strict = true;
  resolution.preferences.compute_family =
      hal::ComputeFamilyAllowance::vulkan_only;
  if (request.request_fixed_center_vignette) {
    resolution.stages[resolution.stage_count++] =
        Intent(kVignetteStageId, hal::SemanticEffectId::vignette);
  }
  if (request.request_mirror) {
    resolution.stages[resolution.stage_count++] =
        Intent(kMirrorStageId, hal::SemanticEffectId::mirror);
  }
  return resolution;
}

} // namespace

struct PreparedHardwareExecutionOwner::Impl {
  hal::StaticAdapterRegistry registry;
  std::unique_ptr<HardwareExecutionRuntime> runtime;
  hal::AdapterRegistration registration{};
  PreparedFinalHardwareRequest active_request{};
  PreparedHardwareProgramIdentity active_program{};
  PreparedHardwareExecutionSnapshot snapshot{};
  std::uint64_t next_program_id = 1;
  hal::StatusId initialization_status = hal::StatusId::unavailable;
  hal::ReasonId initialization_reason =
      hal::ReasonId::capability_evidence_incomplete;

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
  struct Production {
    vulkan::kernels::UtilityKernels kernels;
    std::unique_ptr<vulkan::QualifiedFinalResidentExecutor> executor;
    std::unique_ptr<vulkan::VulkanHalResidentAdapter> adapter;
  };
  std::unique_ptr<Production> production;
#endif

  bool Register(const hal::AdapterRegistration &candidate) noexcept {
    hal::ReasonId reason = hal::ReasonId::none;
    const hal::StatusId status = registry.Register(candidate, &reason);
    initialization_status = status;
    initialization_reason = reason;
    if (status != hal::StatusId::ready)
      return false;
    registration = candidate;
    runtime = std::make_unique<HardwareExecutionRuntime>(registry);
    return true;
  }

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
  static hal::SemanticCapabilityEvidence
  Capability(hal::SemanticEffectId semantic,
             hal::CapabilityEvidenceCode production_code) noexcept {
    hal::SemanticCapabilityEvidence capability;
    capability.semantic = semantic;
    capability.provider =
        hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    capability.status = hal::StatusId::ready;
    capability.reason = hal::ReasonId::none;
    capability.live_stage = true;
    capability.production_eligible = true;
    capability.exact_evidence = true;
    constexpr hal::CapabilityEvidenceCode kCommon[]{
        hal::CapabilityEvidenceCode::vulkan_build_enabled,
        hal::CapabilityEvidenceCode::vulkan_loader_available,
        hal::CapabilityEvidenceCode::vulkan_non_cpu_device,
        hal::CapabilityEvidenceCode::vulkan_compute_queue_available,
        hal::CapabilityEvidenceCode::vulkan_context_healthy,
        hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready,
        hal::CapabilityEvidenceCode::vulkan_exact_device_shared,
        hal::CapabilityEvidenceCode::vulkan_device_local_residency,
        hal::CapabilityEvidenceCode::vulkan_bounded_allocations,
        hal::CapabilityEvidenceCode::vulkan_no_cpu_layers,
        hal::CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
        hal::CapabilityEvidenceCode::vulkan_live_stage_present,
    };
    for (const auto code : kCommon) {
      capability.evidence[capability.evidence_count++] = {
          code,     hal::StatusId::ready,   hal::ReasonId::none,
          semantic, hal::BackendId::vulkan, true};
    }
    capability.evidence[capability.evidence_count++] = {
        production_code, hal::StatusId::ready,   hal::ReasonId::none,
        semantic,        hal::BackendId::vulkan, true};
    return capability;
  }

  bool InitializeProduction() noexcept {
    if (runtime)
      return true;
    production = std::make_unique<Production>();
    std::string error;
    if (!production->kernels.Initialize(&error)) {
      initialization_status = hal::StatusId::unavailable;
      initialization_reason = hal::ReasonId::capability_evidence_incomplete;
      production.reset();
      return false;
    }
    static std::atomic<std::uint64_t> next_adapter_id{UINT64_C(0x56000000)};
    std::uint64_t adapter_id =
        next_adapter_id.fetch_add(1, std::memory_order_relaxed);
    if (adapter_id == 0)
      adapter_id = next_adapter_id.fetch_add(1, std::memory_order_relaxed);
    production->executor =
        std::make_unique<vulkan::QualifiedFinalResidentExecutor>(
            adapter_id, production->kernels);
    const auto evidence = production->executor->CapabilityEvidence();
    if (evidence.blocker != hal::ReasonId::none ||
        !evidence.mirror_production_ready ||
        !evidence.vignette_fixed_center_production_ready) {
      initialization_status = hal::StatusId::unavailable;
      initialization_reason =
          evidence.blocker == hal::ReasonId::none
              ? hal::ReasonId::capability_evidence_incomplete
              : evidence.blocker;
      production.reset();
      return false;
    }
    production->adapter = std::make_unique<vulkan::VulkanHalResidentAdapter>(
        adapter_id, *production->executor);
    hal::AdapterRegistration candidate;
    candidate.adapter_instance_id = adapter_id;
    candidate.backend = hal::BackendId::vulkan;
    candidate.execution_domain = evidence.execution_domain;
    candidate.selection_priority = 300;
    candidate.resident_adapter = production->adapter.get();
    candidate.semantic_capabilities[0] =
        Capability(hal::SemanticEffectId::vignette,
                   hal::CapabilityEvidenceCode::
                       open_vulkan_vignette_fixed_center_production_ready);
    candidate.semantic_capabilities[1] = Capability(
        hal::SemanticEffectId::mirror,
        hal::CapabilityEvidenceCode::open_vulkan_mirror_production_ready);
    candidate.semantic_capability_count = 2;
    return Register(candidate);
  }

  bool
  SetProductionOptions(const hal::PreparedProgram &program,
                       const PreparedFinalHardwareRequest &request) noexcept {
    if (!production || !production->adapter)
      return true;
    vulkan::VulkanResidentProgramOptions options;
    options.program_id = program.program_id;
    options.canonical_plan_hash = program.invalidation_key.canonical_plan_hash;
    options.request_vignette = request.request_fixed_center_vignette;
    options.request_mirror = request.request_mirror;
    options.vignette_center =
        vulkan::VulkanVignetteCenterMode::fixed_frame_center;
    options.vignette_intensity_percent = request.vignette_intensity_percent;
    return production->adapter->SetProgramOptions(options);
  }
#else
  bool InitializeProduction() noexcept { return false; }
  bool SetProductionOptions(const hal::PreparedProgram &,
                            const PreparedFinalHardwareRequest &) noexcept {
    // Neutral injected adapters do not expose backend-specific program
    // options. Their setup behavior is fully described by PreparedProgram.
    return runtime != nullptr;
  }
#endif

  void RefreshSnapshot(PreparedHardwareRoute route,
                       PreparedHardwareEligibility eligibility) noexcept {
    snapshot.route = route;
    snapshot.eligibility = eligibility;
    if (!runtime) {
      snapshot.prepared = false;
      snapshot.active_program = {};
      return;
    }
    snapshot.setup = runtime->latest_setup();
    snapshot.readiness = runtime->readiness();
    snapshot.telemetry = runtime->telemetry();
    snapshot.latest_committed_frame = runtime->latest_committed_frame();
    snapshot.prepared = runtime->prepared();
    snapshot.active_program = snapshot.prepared ? active_program
                                                : PreparedHardwareProgramIdentity{};
  }

  ~Impl() {
    if (runtime)
      runtime->Shutdown();
    runtime.reset();
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (production) {
      production->adapter.reset();
      production->executor.reset();
      production->kernels.Shutdown();
    }
#endif
  }
};

PreparedHardwareCompatibilityTelemetry
ProjectPreparedHardwareCompatibilityTelemetry(
    const PreparedHardwareExecutionSnapshot &snapshot) noexcept {
  PreparedHardwareCompatibilityTelemetry projected;
  projected.active_frames = snapshot.telemetry.active_backend_frames;
  projected.upload_calls = snapshot.telemetry.transfers.uploads;
  projected.download_calls = snapshot.telemetry.transfers.final_readbacks;
  projected.final_download_calls = snapshot.telemetry.transfers.final_readbacks;
  projected.forced_sync_calls = snapshot.telemetry.transfers.synchronizations;
  projected.fallback_frames = snapshot.telemetry.fallback_frames;
  projected.runtime_failure_frames =
      snapshot.telemetry.execution_attempts >=
              snapshot.telemetry.execution_successes
          ? snapshot.telemetry.execution_attempts -
                snapshot.telemetry.execution_successes
          : 0;
  return projected;
}

std::string_view PreparedHardwareReasonCode(hal::ReasonId reason) noexcept {
  switch (reason) {
  case hal::ReasonId::none:
    return "none";
  case hal::ReasonId::optional_stage_failed_open:
    return "optional_stage_failed_open";
  case hal::ReasonId::device_lost:
    return "device_lost";
  case hal::ReasonId::synchronization_failed:
    return "synchronization_failed";
  case hal::ReasonId::circuit_breaker_open:
    return "circuit_breaker_open";
  case hal::ReasonId::reconfiguration_required:
    return "reconfiguration_required";
  case hal::ReasonId::wrong_execution_domain:
    return "wrong_execution_domain";
  case hal::ReasonId::invalid_resource:
    return "invalid_resource";
  case hal::ReasonId::resource_exhausted:
    return "resource_exhausted";
  case hal::ReasonId::output_write_failed:
    return "output_write_failed";
  case hal::ReasonId::stale_program:
    return "stale_program";
  case hal::ReasonId::shutdown_requested:
    return "shutdown_requested";
  default:
    return "hal_contract_failure";
  }
}

PreparedHardwareEligibility EvaluatePreparedFinalHardwareEligibility(
    const PreparedFinalHardwareRequest &request) noexcept {
  if (request.requested_backend != hal::BackendId::vulkan ||
      !request.strict_backend_selection) {
    return PreparedHardwareEligibility::not_strict_explicit_vulkan;
  }
  if (!request.request_fixed_center_vignette && !request.request_mirror)
    return PreparedHardwareEligibility::no_qualified_final_stage;
  if (request.contains_unqualified_stage)
    return PreparedHardwareEligibility::contains_unqualified_stage;
  if (request.tracked_vignette_center)
    return PreparedHardwareEligibility::tracked_vignette_center;
  if (!request.host_input_is_tight_rgb24 ||
      !request.host_output_is_tight_rgb24 || request.input_width == 0 ||
      request.input_height == 0 || request.output_width == 0 ||
      request.output_height == 0 || request.vignette_intensity_percent < 0 ||
      request.vignette_intensity_percent > 100) {
    return PreparedHardwareEligibility::unsupported_host_layout;
  }
  if (request.generations.configuration == 0 ||
      request.generations.geometry == 0) {
    return PreparedHardwareEligibility::invalid_generation;
  }
  return PreparedHardwareEligibility::eligible;
}

PreparedHardwareExecutionOwner::PreparedHardwareExecutionOwner()
    : impl_(std::make_unique<Impl>()) {}

PreparedHardwareExecutionOwner::PreparedHardwareExecutionOwner(
    const hal::AdapterRegistration &registration)
    : impl_(std::make_unique<Impl>()) {
  (void)impl_->Register(registration);
}

PreparedHardwareExecutionOwner::~PreparedHardwareExecutionOwner() = default;

PreparedHardwareSetupResult PreparedHardwareExecutionOwner::ConfigureFinal(
    const PreparedFinalHardwareRequest &request) noexcept {
  PreparedHardwareSetupResult result;
  result.eligibility = EvaluatePreparedFinalHardwareEligibility(request);
  if (result.eligibility != PreparedHardwareEligibility::eligible) {
    if (impl_->runtime && impl_->runtime->prepared())
      impl_->runtime->Invalidate(hal::ReasonId::reconfiguration_required);
    impl_->RefreshSnapshot(PreparedHardwareRoute::legacy_shadow,
                           result.eligibility);
    result.route = PreparedHardwareRoute::legacy_shadow;
    result.status = hal::StatusId::blocked;
    result.reason = result.eligibility ==
                            PreparedHardwareEligibility::tracked_vignette_center
                        ? hal::ReasonId::semantic_effect_unsupported
                        : hal::ReasonId::capability_evidence_incomplete;
    result.setup = impl_->snapshot.setup;
    return result;
  }
  if (!impl_->runtime && !impl_->InitializeProduction()) {
    impl_->RefreshSnapshot(PreparedHardwareRoute::legacy_shadow,
                           PreparedHardwareEligibility::unavailable);
    result.eligibility = PreparedHardwareEligibility::unavailable;
    result.status = impl_->initialization_status;
    result.reason = impl_->initialization_reason;
    return result;
  }

  const std::uint64_t program_id = impl_->next_program_id++;
  const hal::PreparedProgram program =
      Program(request, impl_->registration, program_id);
  hal::ReasonId program_reason = hal::ReasonId::none;
  if (program.program_id == 0 ||
      !hal::ValidatePreparedProgram(program, &program_reason) ||
      !impl_->SetProductionOptions(program, request)) {
    result.status = hal::StatusId::blocked;
    result.reason = program_reason == hal::ReasonId::none
                        ? hal::ReasonId::malformed_contract
                        : program_reason;
  } else {
    HardwareExecutionSetupRequest setup;
    setup.resolution = Resolution(request);
    setup.prepared_program = program;
    setup.mode = HardwareExecutionSetupMode::hal_live;
    // CameraPipeline's retained path has not yet been projected into the
    // neutral resolver schema. Do not present a self-comparison of the HAL
    // registry as independent legacy parity evidence.
    setup.compare_legacy_resolution = false;
    setup.legacy_path_retained_for_setup_rollback = true;
    result.status = impl_->runtime->Configure(setup, &result.reason);
    result.setup = impl_->runtime->latest_setup();
  }

  const bool success = result.status == hal::StatusId::ready ||
                       result.status == hal::StatusId::degraded;
  if (success) {
    impl_->active_request = request;
    impl_->active_request.generations.context =
        impl_->registration.execution_domain.logical_context_generation;
    impl_->active_program.valid = true;
    impl_->active_program.program_id = program.program_id;
    impl_->active_program.owner_adapter_instance_id =
        program.owner_adapter_instance_id;
    impl_->active_program.invalidation_key = program.invalidation_key;
    impl_->active_program.execution_domain = program.execution_domain;
    impl_->active_program.stage_count = program.stage_count;
    result.route = PreparedHardwareRoute::hal_live;
  } else if (impl_->runtime->prepared() &&
             SameLiveProgram(impl_->active_request, request)) {
    result.route = PreparedHardwareRoute::hal_live;
    result.retained_previous_live_program = true;
  } else {
    if (impl_->runtime->prepared())
      impl_->runtime->Invalidate(hal::ReasonId::reconfiguration_required);
    result.route = PreparedHardwareRoute::legacy_shadow;
  }
  impl_->RefreshSnapshot(result.route, result.eligibility);
  result.setup = impl_->snapshot.setup;
  return result;
}

PreparedHardwareFrameResult PreparedHardwareExecutionOwner::ExecuteHostFrame(
    const std::uint8_t *input, std::size_t input_size_bytes,
    std::size_t input_row_stride_bytes, std::uint8_t *output,
    std::size_t output_size_bytes, std::size_t output_row_stride_bytes,
    std::uint64_t capture_sequence) noexcept {
  PreparedHardwareFrameResult result;
  if (!impl_->runtime || !impl_->runtime->prepared() || !input || !output) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::stale_program;
    return result;
  }
  hal::HostFrameBinding binding;
  const auto &request = impl_->active_request;
  binding.input_descriptor =
      Image(request.input_width, request.input_height,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  binding.output_descriptor =
      Image(request.output_width, request.output_height,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = capture_sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input);
  binding.input_planes[0].span_bytes = input_size_bytes;
  binding.input_planes[0].row_stride_bytes = input_row_stride_bytes;
  binding.output_planes[0].write_data = reinterpret_cast<std::byte *>(output);
  binding.output_planes[0].span_bytes = output_size_bytes;
  binding.output_planes[0].row_stride_bytes = output_row_stride_bytes;

  const HardwareFrameResult frame = impl_->runtime->ExecuteHostFrame(binding);
  result.status = frame.status;
  result.reason = frame.reason;
  result.completion = frame.completion;
  result.output_ready = frame.output_ready;
  result.completion_pending = impl_->runtime->frame_pending();
  if (frame.output_ready && frame.output.plane_count == 1) {
    const auto &plane = frame.output.planes[0];
    result.output = reinterpret_cast<const std::uint8_t *>(plane.write_data);
    result.output_size_bytes = static_cast<std::size_t>(plane.span_bytes);
    result.output_row_stride_bytes =
        static_cast<std::size_t>(plane.row_stride_bytes);
  }
  impl_->RefreshSnapshot(PreparedHardwareRoute::hal_live,
                         PreparedHardwareEligibility::eligible);
  return result;
}

hal::StatusId PreparedHardwareExecutionOwner::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  if (!impl_->runtime) {
    if (reason)
      *reason = hal::ReasonId::stale_program;
    return hal::StatusId::blocked;
  }
  const hal::StatusId status =
      impl_->runtime->CommitOutput(completion, output_write_succeeded, reason);
  impl_->RefreshSnapshot(impl_->snapshot.route, impl_->snapshot.eligibility);
  return status;
}

void PreparedHardwareExecutionOwner::Invalidate(hal::ReasonId reason) noexcept {
  if (impl_->runtime)
    impl_->runtime->Invalidate(reason);
  impl_->RefreshSnapshot(PreparedHardwareRoute::legacy_shadow,
                         PreparedHardwareEligibility::unavailable);
}

void PreparedHardwareExecutionOwner::Shutdown() noexcept {
  if (impl_->runtime)
    impl_->runtime->Shutdown();
  impl_->RefreshSnapshot(PreparedHardwareRoute::legacy_shadow,
                         PreparedHardwareEligibility::unavailable);
}

PreparedHardwareExecutionSnapshot
PreparedHardwareExecutionOwner::Snapshot() const noexcept {
  return impl_->snapshot;
}

const PreparedFinalHardwareRequest &
PreparedHardwareExecutionOwner::active_request() const noexcept {
  return impl_->active_request;
}

} // namespace studiocast::video
