#include "core/video/prepared_maxine_hardware_execution_owner.h"

#include "core/cuda/cuda_driver_api.h"
#include "core/maxine/production_hal_owner.h"

#include <atomic>
#include <limits>
#include <new>
#include <utility>

namespace studiocast::video {
namespace {

constexpr std::uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kHashPrime = UINT64_C(1099511628211);

std::uint64_t NextAdapterInstanceId() noexcept {
  static std::atomic<std::uint64_t> next{UINT64_C(0x4d580000)};
  std::uint64_t id = next.fetch_add(1, std::memory_order_relaxed);
  if (id == 0)
    id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
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

bool IsZeroBudget(const hal::ResourceBudget &budget) noexcept {
  return budget.persistent_bytes == 0 && budget.scratch_bytes == 0 &&
         budget.upload_staging_bytes == 0 &&
         budget.readback_staging_bytes == 0 && budget.resource_count == 0;
}

std::uint64_t HashValue(std::uint64_t hash, std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    hash ^= static_cast<std::uint8_t>(value >> shift);
    hash *= kHashPrime;
  }
  return hash;
}

std::uint64_t HashText(std::uint64_t hash, std::string_view text) noexcept {
  for (const char value : text) {
    hash ^= static_cast<std::uint8_t>(value);
    hash *= kHashPrime;
  }
  return hash;
}

hal::SemanticEffectProviderId
ProviderFor(hal::SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case hal::SemanticEffectId::eye_contact:
  case hal::SemanticEffectId::auto_frame:
    return hal::SemanticEffectProviderId::maxine_ar;
  case hal::SemanticEffectId::video_noise_removal:
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
  case hal::SemanticEffectId::virtual_key_light:
    return hal::SemanticEffectProviderId::maxine_vfx;
  case hal::SemanticEffectId::vignette:
    return hal::SemanticEffectProviderId::maxine_cuda_utility;
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return hal::SemanticEffectProviderId::invalid;
  }
  return hal::SemanticEffectProviderId::invalid;
}

bool ProductionStageKind(hal::SemanticEffectId semantic,
                         maxine::ResidentStageKind *kind) noexcept {
  if (!kind)
    return false;
  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
    *kind = maxine::ResidentStageKind::denoise;
    return true;
  case hal::SemanticEffectId::eye_contact:
    *kind = maxine::ResidentStageKind::eye_contact;
    return true;
  case hal::SemanticEffectId::virtual_background_blur:
    *kind = maxine::ResidentStageKind::background_blur;
    return true;
  case hal::SemanticEffectId::virtual_background_remove:
    *kind = maxine::ResidentStageKind::background_remove;
    return true;
  case hal::SemanticEffectId::virtual_background_replace:
    *kind = maxine::ResidentStageKind::background_replace;
    return true;
  case hal::SemanticEffectId::virtual_key_light:
    *kind = maxine::ResidentStageKind::relighting;
    return true;
  case hal::SemanticEffectId::auto_frame:
    *kind = maxine::ResidentStageKind::auto_frame;
    return true;
  case hal::SemanticEffectId::vignette:
    *kind = maxine::ResidentStageKind::vignette;
    return true;
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

bool IsQualifiedV1Stage(hal::SemanticEffectId semantic) noexcept {
  return semantic == hal::SemanticEffectId::virtual_background_blur ||
         semantic == hal::SemanticEffectId::virtual_background_remove ||
         semantic == hal::SemanticEffectId::virtual_background_replace ||
         semantic == hal::SemanticEffectId::virtual_key_light ||
         semantic == hal::SemanticEffectId::vignette;
}

bool HasIdentityBytes(const std::array<std::uint8_t, 32> &identity) noexcept {
  for (const auto byte : identity) {
    if (byte != 0)
      return true;
  }
  return false;
}

bool ValidModelArtifactEvidence(
    const PreparedMaxineHardwareStageRequest &stage) noexcept {
  if (stage.semantic == hal::SemanticEffectId::vignette)
    return true;
  return stage.model.artifact_generation != 0 &&
         HasIdentityBytes(stage.model.model_id_hash) &&
         HasIdentityBytes(stage.model.artifact_set_hash);
}

hal::ImageDescriptor Image(std::uint32_t width, std::uint32_t height,
                           hal::PixelFormat format,
                           hal::ChannelOrder channel_order,
                           hal::HostCoherence coherence,
                           hal::Mutability mutability) noexcept {
  hal::ImageDescriptor image;
  image.width = width;
  image.height = height;
  image.plane_count = 1;
  image.format = format;
  image.channel_order = channel_order;
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

hal::ResourceBudget DefaultInjectedBudget(
    const PreparedMaxineHardwareRequest &request) noexcept {
  std::uint64_t bytes = 0;
  if (!Multiply(request.width, request.height, &bytes) ||
      !Multiply(bytes, 3, &bytes) ||
      bytes > std::numeric_limits<std::uint64_t>::max() / 8u) {
    return {};
  }
  hal::ResourceBudget budget;
  budget.persistent_bytes = bytes * 8u;
  budget.scratch_bytes = bytes;
  budget.upload_staging_bytes = bytes;
  budget.readback_staging_bytes = bytes;
  budget.resource_count = 16;
  budget.in_flight_slots = 1;
  return budget;
}

std::uint64_t HashRequest(const PreparedMaxineHardwareRequest &request) noexcept {
  std::uint64_t hash = kHashOffset;
  hash = HashValue(hash, request.width);
  hash = HashValue(hash, request.height);
  hash = HashValue(hash, request.stage_count);
  hash = HashValue(
      hash, static_cast<std::uint64_t>(
                request.effects.virtual_background.strength));
  hash = HashValue(
      hash, static_cast<std::uint64_t>(request.effects.virtual_background.mode));
  hash = HashValue(
      hash,
      static_cast<std::uint64_t>(
          request.effects.virtual_key_light.intensity));
  hash = HashValue(
      hash, static_cast<std::uint64_t>(request.effects.vignette.intensity));
  hash = HashText(hash, request.vfx_model_directory);
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    hash = HashValue(hash, request.stages[index].stage_id);
    hash = HashValue(hash,
                     static_cast<std::uint64_t>(request.stages[index].semantic));
  }
  return hash == 0 ? 1 : hash;
}

std::uint64_t HashDomain(const hal::ExecutionDomainId &domain) noexcept {
  std::uint64_t hash = kHashOffset;
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
  hash = HashValue(
      hash, static_cast<std::uint64_t>(domain.synchronization_protocol));
  hash = HashValue(hash, domain.adapter_instance_id);
  return hash == 0 ? 1 : hash;
}

hal::ProgramGenerations AuthoritativeGenerations(
    const PreparedMaxineHardwareRequest &request,
    const hal::ExecutionDomainId &domain) noexcept {
  hal::ProgramGenerations generations;
  generations.configuration = request.generations.configuration;
  generations.geometry =
      HashValue(HashValue(kHashOffset, request.width), request.height);
  std::uint64_t model = kHashOffset;
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    model = HashValue(
        model, static_cast<std::uint64_t>(request.stages[index].semantic));
    model = HashValue(model,
                      request.stages[index].model.artifact_generation);
    for (const auto byte : request.stages[index].model.model_id_hash)
      model = HashValue(model, byte);
    for (const auto byte : request.stages[index].model.artifact_set_hash)
      model = HashValue(model, byte);
  }
  generations.model =
      HashValue(model, request.replacement.generation);
  std::uint64_t provider = kHashOffset;
  provider = HashValue(provider, request.runtime.vfx);
  provider = HashValue(provider, request.runtime.nvcv);
  provider = HashValue(provider, request.runtime.ar);
  provider = HashValue(provider, request.runtime.cuda);
  provider = HashValue(provider, static_cast<std::uint64_t>(domain.api_family));
  generations.provider = provider == 0 ? 1 : provider;
  std::uint64_t device = kHashOffset;
  device = HashValue(device, domain.physical_device.vendor_id);
  device = HashValue(device, domain.physical_device.device_id);
  for (const auto byte : domain.physical_device.stable_id)
    device = HashValue(device, byte);
  generations.device = device == 0 ? 1 : device;
  generations.context = domain.logical_context_generation;
  if (generations.geometry == 0)
    generations.geometry = 1;
  if (generations.model == 0)
    generations.model = 1;
  return generations;
}

std::uint64_t ValidatedArtifactSetHash(
    const PreparedMaxineHardwareRequest &request) noexcept {
  std::uint64_t hash = kHashOffset;
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    const auto &stage = request.stages[index];
    hash = HashValue(hash, static_cast<std::uint64_t>(stage.semantic));
    hash = HashValue(hash, stage.model.artifact_generation);
    for (const auto byte : stage.model.model_id_hash)
      hash = HashValue(hash, byte);
    for (const auto byte : stage.model.artifact_set_hash)
      hash = HashValue(hash, byte);
  }
  return HashValue(hash, request.replacement.generation);
}

hal::PreparedProgram Program(const PreparedMaxineHardwareRequest &request,
                             const hal::ExecutionDomainId &domain,
                             const hal::ResourceBudget &budget,
                             std::uint64_t program_id) noexcept {
  hal::PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = domain.adapter_instance_id;
  program.execution_domain = domain;
  program.host_input =
      Image(request.width, request.height, hal::PixelFormat::rgb_u8,
            hal::ChannelOrder::rgb, hal::HostCoherence::coherent,
            hal::Mutability::read_only);
  program.host_output =
      Image(request.width, request.height, hal::PixelFormat::bgr_u8,
            hal::ChannelOrder::bgr, hal::HostCoherence::coherent,
            hal::Mutability::write_only);
  program.invalidation_key.generations = request.generations;
  program.invalidation_key.generations.context =
      domain.logical_context_generation;
  program.invalidation_key.canonical_plan_hash = HashRequest(request);
  program.invalidation_key.model_artifact_hash =
      ValidatedArtifactSetHash(request);
  if (program.invalidation_key.model_artifact_hash == 0)
    program.invalidation_key.model_artifact_hash = 1;
  program.invalidation_key.execution_domain_hash = HashDomain(domain);
  program.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  program.requested_backend = hal::BackendId::maxine;
  program.strict_backend_selection = true;
  program.requested_execution_domain = domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;
  program.budget = budget;

  const auto resident_input =
      Image(request.width, request.height, hal::PixelFormat::bgr_u8,
            hal::ChannelOrder::bgr, hal::HostCoherence::not_host_visible,
            hal::Mutability::read_only);
  const auto resident_output =
      Image(request.width, request.height, hal::PixelFormat::bgr_u8,
            hal::ChannelOrder::bgr, hal::HostCoherence::not_host_visible,
            hal::Mutability::read_write);
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    const auto &intent = request.stages[index];
    auto &stage = program.stages[program.stage_count++];
    stage.stage_id = intent.stage_id;
    stage.semantic = intent.semantic;
    stage.provider = ProviderFor(intent.semantic);
    stage.backend = hal::BackendId::maxine;
    stage.execution_domain = domain;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = resident_input;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = resident_output;
    stage.required = intent.required;
    stage.failure_action =
        intent.required
            ? hal::FailureAction::fail_closed
            : hal::FailureAction::fail_open_preserve_last_valid;
    stage.circuit_breaker.consecutive_failure_limit = 1;
    stage.circuit_breaker.cooldown_frames = 30;
    stage.circuit_breaker.action =
        intent.required ? hal::CircuitBreakerAction::quarantine_program
                        : hal::CircuitBreakerAction::disable_stage;
    stage.model = intent.model;
  }
  if (request.declares_output_format_cpu_tail && program.stage_count != 0) {
    auto &tail = program.cpu_tails[program.cpu_tail_count++];
    tail.id = hal::CpuTailId::output_format_conversion;
    tail.placement = hal::DeclaredCpuTail::Placement::after_stage;
    tail.after_stage_id = program.stages[program.stage_count - 1].stage_id;
    tail.boundary_index = static_cast<std::uint32_t>(program.stage_count);
  }
  return program;
}

maxine::ProductionResidentSetup
ProductionSetup(const PreparedMaxineHardwareRequest &request,
                std::uint32_t stage_mask) {
  maxine::ProductionResidentSetup setup;
  setup.configuration_generation = request.generations.configuration;
  setup.width = request.width;
  setup.height = request.height;
  setup.output_width = request.width;
  setup.output_height = request.height;
  setup.enabled_maxine_stage_mask = stage_mask;
  setup.vfx_model_directory = request.vfx_model_directory;
  setup.effects = request.effects;
  setup.vignette_center_x_px = request.vignette_center_x_px;
  setup.vignette_center_y_px = request.vignette_center_y_px;
  setup.replacement_background = {
      request.replacement.rgb,
      request.replacement.width,
      request.replacement.height,
      request.replacement.stride_bytes,
      request.replacement.generation,
  };
  return setup;
}

bool Delta(std::uint64_t before, std::uint64_t after,
           std::uint64_t *value) noexcept {
  if (!value || after < before)
    return false;
  *value = after - before;
  return true;
}

PreparedMaxineCompatibilityTelemetry CompatibilityDelta(
    const maxine::ProductionResidentTelemetry &before,
    const maxine::ProductionResidentTelemetry &after,
    const PreparedMaxineHardwareRequest &request) noexcept {
  PreparedMaxineCompatibilityTelemetry result;
  const auto transfer = [&](maxine::ProductionTransferKind kind,
                            std::uint64_t *attempts,
                            std::uint64_t *successes) {
    const auto index = static_cast<std::size_t>(kind);
    return Delta(before.transfers[index].attempts,
                 after.transfers[index].attempts, attempts) &&
           Delta(before.transfers[index].successes,
                 after.transfers[index].successes, successes);
  };
  const auto cpu_index = static_cast<std::size_t>(
      maxine::ProductionCpuStageKind::rgb_to_bgr_staging);
  if (!Delta(before.cpu_stages[cpu_index].successes,
             after.cpu_stages[cpu_index].successes,
             &result.rgb_to_bgr_calls) ||
      !transfer(maxine::ProductionTransferKind::host_upload,
                &result.upload_attempts, &result.upload_calls) ||
      !transfer(maxine::ProductionTransferKind::final_download,
                &result.final_download_attempts,
                &result.final_download_calls) ||
      !transfer(maxine::ProductionTransferKind::cpu_continuation_download,
                &result.cpu_continuation_download_attempts,
                &result.cpu_continuation_download_calls) ||
      !transfer(maxine::ProductionTransferKind::device_format_bridge,
                &result.device_bridge_attempts,
                &result.device_bridge_calls) ||
      !Delta(before.matte_inferences.attempts,
             after.matte_inferences.attempts,
             &result.matte_inference_attempts) ||
      !Delta(before.matte_inferences.successes,
             after.matte_inferences.successes,
             &result.matte_inference_calls) ||
      !Delta(before.shared_matte_reuses, after.shared_matte_reuses,
             &result.shared_matte_reuses) ||
      !Delta(before.explicit_synchronizations.attempts,
             after.explicit_synchronizations.attempts,
             &result.sync_attempts) ||
      !Delta(before.explicit_synchronizations.successes,
             after.explicit_synchronizations.successes,
             &result.sync_calls) ||
      !Delta(before.composites.attempts, after.composites.attempts,
             &result.composite_attempts) ||
      !Delta(before.composites.successes, after.composites.successes,
             &result.composite_calls) ||
      !Delta(before.synchronous_sdk_runs.attempts,
             after.synchronous_sdk_runs.attempts,
             &result.synchronous_sdk_run_attempts) ||
      !Delta(before.synchronous_sdk_runs.successes,
             after.synchronous_sdk_runs.successes,
             &result.synchronous_sdk_run_calls) ||
      !Delta(before.asynchronous_sdk_runs.attempts,
             after.asynchronous_sdk_runs.attempts,
             &result.asynchronous_sdk_run_attempts) ||
      !Delta(before.asynchronous_sdk_runs.successes,
             after.asynchronous_sdk_runs.successes,
             &result.asynchronous_sdk_run_calls)) {
    return {};
  }
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    maxine::ResidentStageKind kind{};
    if (!ProductionStageKind(request.stages[index].semantic, &kind))
      return {};
    const auto stage_index = static_cast<std::size_t>(kind);
    if (!Delta(before.stages[stage_index].attempts,
               after.stages[stage_index].attempts,
               &result.stage_attempts[index]) ||
        !Delta(before.stages[stage_index].successes,
               after.stages[stage_index].successes,
               &result.stage_successes[index])) {
      return {};
    }
  }
  return result;
}

} // namespace

PreparedMaxineHardwareEligibility EvaluatePreparedMaxineHardwareEligibility(
    const PreparedMaxineHardwareRequest &request) noexcept {
  if (request.requested_backend != hal::BackendId::maxine ||
      !request.strict_backend_selection) {
    return PreparedMaxineHardwareEligibility::not_exact_maxine;
  }
  if (!request.single_resident_island)
    return PreparedMaxineHardwareEligibility::multiple_resident_islands;
  if (request.contains_other_execution_domain)
    return PreparedMaxineHardwareEligibility::cross_domain_chain;
  if (request.stage_count == 0)
    return PreparedMaxineHardwareEligibility::no_qualified_stage;
  if (request.stage_count > request.stages.size())
    return PreparedMaxineHardwareEligibility::unsupported_stage;
  if (request.temporal_reset_required)
    return PreparedMaxineHardwareEligibility::temporal_reset_unsupported;
  if (!request.host_input_is_tight_rgb24 ||
      !request.host_output_is_tight_bgr24 || request.width == 0 ||
      request.height == 0) {
    return PreparedMaxineHardwareEligibility::unsupported_host_layout;
  }
  if (request.generations.configuration == 0) {
    return PreparedMaxineHardwareEligibility::invalid_generation;
  }
  std::uint32_t seen = 0;
  std::uint32_t previous_semantic = 0;
  bool has_virtual_background = false;
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    const auto &stage = request.stages[index];
    const auto semantic = static_cast<std::uint32_t>(stage.semantic);
    const bool virtual_background =
        stage.semantic == hal::SemanticEffectId::virtual_background_blur ||
        stage.semantic == hal::SemanticEffectId::virtual_background_remove ||
        stage.semantic == hal::SemanticEffectId::virtual_background_replace;
    if (stage.stage_id == 0 || !IsQualifiedV1Stage(stage.semantic) ||
        ProviderFor(stage.semantic) ==
            hal::SemanticEffectProviderId::invalid ||
        semantic >= 32 || semantic <= previous_semantic ||
        (seen & (std::uint32_t{1} << semantic)) != 0 ||
        (virtual_background && has_virtual_background)) {
      return PreparedMaxineHardwareEligibility::unsupported_stage;
    }
    if (!ValidModelArtifactEvidence(stage)) {
      return PreparedMaxineHardwareEligibility::
          model_artifact_evidence_incomplete;
    }
    seen |= std::uint32_t{1} << semantic;
    previous_semantic = semantic;
    has_virtual_background |= virtual_background;
  }
  return PreparedMaxineHardwareEligibility::eligible;
}

std::string_view
PreparedMaxineHardwareReasonCode(hal::ReasonId reason) noexcept {
  switch (reason) {
  case hal::ReasonId::none:
    return "none";
  case hal::ReasonId::resource_exhausted:
    return "resource_exhausted";
  case hal::ReasonId::wrong_execution_domain:
    return "wrong_execution_domain";
  case hal::ReasonId::wrong_generation:
    return "wrong_generation";
  case hal::ReasonId::foreign_adapter:
    return "foreign_adapter";
  case hal::ReasonId::stale_program:
  case hal::ReasonId::reconfiguration_required:
    return "stale_program";
  case hal::ReasonId::device_lost:
    return "device_lost";
  case hal::ReasonId::optional_stage_failed_open:
    return "optional_stage_failed_open";
  case hal::ReasonId::output_write_failed:
  case hal::ReasonId::output_not_committed:
    return "output_not_committed";
  case hal::ReasonId::shutdown_requested:
    return "shutdown_requested";
  default:
    return "maxine_hal_contract_failure";
  }
}

struct PreparedMaxineHardwareExecutionOwner::Impl {
  const hal::AdapterRegistration *injected_registration = nullptr;
  std::unique_ptr<hal::PreparedProgramSession> injected_session;
  std::unique_ptr<maxine::MaxineProductionHalOwner> production;
  PreparedMaxineHardwareRequest active_request{};
  PreparedMaxineHardwareExecutionSnapshot snapshot{};
  std::uint64_t next_program_id = 1;
  const std::uint64_t stable_adapter_instance_id = NextAdapterInstanceId();

  [[nodiscard]] hal::PreparedProgramSession *Session() noexcept {
    return production ? production->session() : injected_session.get();
  }
  [[nodiscard]] const hal::PreparedProgramSession *Session() const noexcept {
    return production ? production->session() : injected_session.get();
  }
  [[nodiscard]] hal::IResidentAdapter *InjectedAdapter() const noexcept {
    return injected_registration ? injected_registration->resident_adapter
                                 : nullptr;
  }

  void RefreshSnapshot(PreparedMaxineHardwareRoute route,
                       PreparedMaxineHardwareEligibility eligibility) noexcept {
    snapshot.route = route;
    snapshot.eligibility = eligibility;
    const auto *session = Session();
    if (!session) {
      snapshot.prepared = false;
      return;
    }
    snapshot.readiness = session->readiness();
    snapshot.telemetry = session->telemetry();
    snapshot.latest_committed_frame = session->latest_committed_frame();
    snapshot.active_program = session->program();
    snapshot.execution_domain = snapshot.active_program.execution_domain;
    snapshot.prepared = session->prepared();
    snapshot.quarantined = session->quarantined();
    snapshot.stopped = session->stopped();
  }

  ~Impl() {
    if (auto *session = Session(); session && !session->stopped())
      session->Shutdown();
    production.reset();
    injected_session.reset();
  }
};

PreparedMaxineHardwareExecutionOwner::
    PreparedMaxineHardwareExecutionOwner()
    : impl_(std::make_unique<Impl>()) {}

PreparedMaxineHardwareExecutionOwner::PreparedMaxineHardwareExecutionOwner(
    const hal::AdapterRegistration &registration)
    : impl_(std::make_unique<Impl>()) {
  impl_->injected_registration = &registration;
  impl_->injected_session = std::make_unique<hal::PreparedProgramSession>();
}

PreparedMaxineHardwareExecutionOwner::
    ~PreparedMaxineHardwareExecutionOwner() = default;

PreparedMaxineHardwareSetupResult
PreparedMaxineHardwareExecutionOwner::Configure(
    const PreparedMaxineHardwareRequest &request) noexcept {
  PreparedMaxineHardwareSetupResult result;
  PreparedMaxineHardwareRequest candidate_request = request;
  const auto eligibility =
      EvaluatePreparedMaxineHardwareEligibility(request);
  result.eligibility = eligibility;
  if (eligibility != PreparedMaxineHardwareEligibility::eligible) {
    result.status = hal::StatusId::blocked;
    result.reason =
        eligibility ==
                    PreparedMaxineHardwareEligibility::
                        temporal_reset_unsupported ||
                eligibility ==
                    PreparedMaxineHardwareEligibility::
                        model_artifact_evidence_incomplete
            ? hal::ReasonId::capability_evidence_incomplete
            : hal::ReasonId::semantic_effect_unsupported;
    impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::legacy_compatibility,
                           eligibility);
    return result;
  }

  try {
    hal::ExecutionDomainId domain;
    hal::ResourceBudget budget = request.capacity;
    std::unique_ptr<maxine::MaxineProductionHalOwner> candidate;
    maxine::MaxineProductionPreparationResult preparation;

    if (impl_->injected_registration) {
      const auto &registration = *impl_->injected_registration;
      candidate_request.adapter_instance_id =
          registration.adapter_instance_id;
      if (registration.backend != hal::BackendId::maxine ||
          registration.resident_adapter == nullptr ||
          registration.execution_domain.backend != hal::BackendId::maxine ||
          registration.execution_domain.adapter_instance_id !=
              registration.adapter_instance_id ||
          !hal::ValidateExecutionDomain(registration.execution_domain)) {
        result.status = hal::StatusId::blocked;
        result.reason = hal::ReasonId::wrong_execution_domain;
        impl_->RefreshSnapshot(
            PreparedMaxineHardwareRoute::legacy_compatibility,
            PreparedMaxineHardwareEligibility::runtime_unavailable);
        return result;
      }
      domain = registration.execution_domain;
      if (IsZeroBudget(budget))
        budget = DefaultInjectedBudget(candidate_request);
    } else {
      candidate_request.adapter_instance_id =
          impl_->stable_adapter_instance_id;
      maxine::ProductionResidentRuntime runtime;
      runtime.vfx = reinterpret_cast<maxine::vfx::VfxApi *>(
          candidate_request.runtime.vfx);
      runtime.nvcv =
          reinterpret_cast<maxine::NvcvApi *>(candidate_request.runtime.nvcv);
      runtime.ar =
          reinterpret_cast<maxine::ar::ArApi *>(candidate_request.runtime.ar);
      runtime.cuda =
          reinterpret_cast<cuda::CudaDriverApi *>(
              candidate_request.runtime.cuda);

      std::uint32_t stage_mask = 0;
      for (std::size_t index = 0; index < candidate_request.stage_count;
           ++index) {
        maxine::ResidentStageKind kind{};
        if (!ProductionStageKind(candidate_request.stages[index].semantic,
                                 &kind)) {
          result.status = hal::StatusId::blocked;
          result.reason = hal::ReasonId::semantic_effect_unsupported;
          return result;
        }
        stage_mask |= maxine::ProductionResidentStageBit(kind);
      }
      const auto setup = ProductionSetup(candidate_request, stage_mask);
      if (IsZeroBudget(budget) &&
          !maxine::KnownMinimumMaxineBudget(setup, &budget)) {
        result.status = hal::StatusId::blocked;
        result.reason = hal::ReasonId::malformed_descriptor;
        return result;
      }
      maxine::MaxineProductionBackendRequest backend_request;
      backend_request.adapter_instance_id =
          candidate_request.adapter_instance_id;
      backend_request.runtime = runtime;
      backend_request.setup = setup;
      backend_request.capacity = budget;
      // A null device-ops handle deliberately selects the production
      // executor's internally owned CUDA crop/resize/vignette operations.
      // Tests may inject an exact fake through this setup-only seam.
      backend_request.device_ops =
          reinterpret_cast<maxine::IResidentDeviceOps *>(
              candidate_request.runtime.device_ops);
      candidate = maxine::MaxineProductionHalOwner::CreateBackend(
          backend_request, &preparation);
      if (!candidate) {
        result.status = preparation.status;
        result.reason = preparation.reason;
        result.known_minimum_budget = preparation.known_minimum_budget;
        result.retained_previous_live_program =
            impl_->production && impl_->production->prepared();
        impl_->RefreshSnapshot(
            PreparedMaxineHardwareRoute::legacy_compatibility,
            PreparedMaxineHardwareEligibility::runtime_unavailable);
        return result;
      }
      domain = candidate->execution_domain();
      result.known_minimum_budget = candidate->known_minimum_budget();
    }

    candidate_request.generations =
        AuthoritativeGenerations(candidate_request, domain);
    auto program =
        Program(candidate_request, domain, budget, impl_->next_program_id++);
    if (program.program_id == 0)
      program.program_id = impl_->next_program_id++;

    hal::StatusId status = hal::StatusId::blocked;
    hal::ReasonId reason = hal::ReasonId::none;
    if (candidate) {
      status = candidate->ConfigureProgram(program, &preparation);
      reason = preparation.reason;
    } else {
      hal::ReadinessSnapshot readiness;
      status = impl_->injected_session->Configure(
          program, *impl_->InjectedAdapter(), &readiness, &reason);
    }
    if (status != hal::StatusId::ready &&
        status != hal::StatusId::degraded) {
      result.status = status;
      result.reason = reason;
      result.execution_domain = domain;
      result.retained_previous_live_program =
          impl_->Session() && impl_->Session()->prepared();
      impl_->RefreshSnapshot(
          PreparedMaxineHardwareRoute::legacy_compatibility,
          PreparedMaxineHardwareEligibility::runtime_unavailable);
      return result;
    }

    if (candidate)
      impl_->production = std::move(candidate);
    impl_->active_request = std::move(candidate_request);
    impl_->snapshot.execution_domain = domain;
    result.route = PreparedMaxineHardwareRoute::hal_live;
    result.status = status;
    result.reason = reason;
    result.execution_domain = domain;
    impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::hal_live, eligibility);
    return result;
  } catch (const std::bad_alloc &) {
    result.status = hal::StatusId::resource_exhausted;
    result.reason = hal::ReasonId::resource_exhausted;
  } catch (...) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::capability_evidence_incomplete;
  }
  impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::legacy_compatibility,
                         PreparedMaxineHardwareEligibility::runtime_unavailable);
  return result;
}

PreparedMaxineHardwareFrameResult
PreparedMaxineHardwareExecutionOwner::ExecuteHostFrame(
    const std::uint8_t *input, std::size_t input_size_bytes,
    std::size_t input_row_stride_bytes, std::uint8_t *output_bgr,
    std::size_t output_size_bytes, std::size_t output_row_stride_bytes,
    std::uint64_t capture_sequence) noexcept {
  PreparedMaxineHardwareFrameResult frame;
  auto *session = impl_->Session();
  if (!session || !session->prepared()) {
    frame.status = hal::StatusId::stale;
    frame.reason = hal::ReasonId::stale_program;
    return frame;
  }
  const auto &program = session->program();
  const std::uint64_t expected_bytes =
      program.host_input.planes[0].size_bytes;
  const std::uint64_t expected_stride =
      program.host_input.planes[0].row_stride_bytes;
  if (!input || !output_bgr || capture_sequence == 0 ||
      input_size_bytes != expected_bytes ||
      output_size_bytes != program.host_output.planes[0].size_bytes ||
      input_row_stride_bytes != expected_stride ||
      output_row_stride_bytes !=
          program.host_output.planes[0].row_stride_bytes) {
    frame.status = hal::StatusId::blocked;
    frame.reason = hal::ReasonId::host_frame_binding_invalid;
    return frame;
  }

  maxine::ProductionResidentTelemetry before;
  bool have_production_telemetry = false;
  if (impl_->production && impl_->production->executor()) {
    before = impl_->production->executor()->telemetry();
    have_production_telemetry = true;
  }

  hal::HostFrameBinding binding;
  binding.input_descriptor = program.host_input;
  binding.output_descriptor = program.host_output;
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input);
  binding.input_planes[0].span_bytes = input_size_bytes;
  binding.input_planes[0].row_stride_bytes = input_row_stride_bytes;
  binding.output_planes[0].write_data =
      reinterpret_cast<std::byte *>(output_bgr);
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
  frame.stage_count = execution.stage_count;
  frame.stages = execution.stages;
  frame.completion_pending = execution.completion.complete;
  if (!execution.output_valid || !frame.completion_pending) {
    impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::hal_live,
                           PreparedMaxineHardwareEligibility::eligible);
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
          reinterpret_cast<std::byte *>(output_bgr)) {
    hal::ReasonId commit_reason = hal::ReasonId::none;
    (void)session->CommitOutput({execution.completion, false}, &commit_reason);
    frame.completion_pending = false;
    frame.output_ready = false;
    impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::hal_live,
                           PreparedMaxineHardwareEligibility::eligible);
    return frame;
  }
  frame.status = execution.status;
  frame.reason = execution.reason;
  frame.output = output_bgr;
  frame.output_size_bytes = output_size_bytes;
  frame.output_row_stride_bytes = output_row_stride_bytes;
  frame.output_ready = true;

  if (have_production_telemetry && impl_->production->executor()) {
    frame.compatibility = CompatibilityDelta(
        before, impl_->production->executor()->telemetry(),
        impl_->active_request);
  }
  impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::hal_live,
                         PreparedMaxineHardwareEligibility::eligible);
  return frame;
}

hal::StatusId PreparedMaxineHardwareExecutionOwner::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  auto *session = impl_->Session();
  if (!session) {
    if (reason)
      *reason = hal::ReasonId::stale_program;
    return hal::StatusId::stale;
  }
  const auto status =
      session->CommitOutput({completion, output_write_succeeded}, reason);
  impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::hal_live,
                         PreparedMaxineHardwareEligibility::eligible);
  return status;
}

void PreparedMaxineHardwareExecutionOwner::NotifyDeviceLost() noexcept {
  if (impl_->production)
    impl_->production->NotifyDeviceLost();
  else if (impl_->injected_session)
    impl_->injected_session->Invalidate(hal::ReasonId::device_lost);
  impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::hal_live,
                         PreparedMaxineHardwareEligibility::eligible);
}

void PreparedMaxineHardwareExecutionOwner::Invalidate(
    hal::ReasonId reason) noexcept {
  if (auto *session = impl_->Session())
    session->Invalidate(reason);
  impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::legacy_compatibility,
                         PreparedMaxineHardwareEligibility::runtime_unavailable);
}

void PreparedMaxineHardwareExecutionOwner::Shutdown() noexcept {
  if (auto *session = impl_->Session(); session && !session->stopped())
    session->Shutdown();
  impl_->production.reset();
  impl_->RefreshSnapshot(PreparedMaxineHardwareRoute::legacy_compatibility,
                         PreparedMaxineHardwareEligibility::runtime_unavailable);
  impl_->snapshot.prepared = false;
  impl_->snapshot.stopped = true;
}

PreparedMaxineHardwareExecutionSnapshot
PreparedMaxineHardwareExecutionOwner::Snapshot() const noexcept {
  return impl_->snapshot;
}

const PreparedMaxineHardwareRequest &
PreparedMaxineHardwareExecutionOwner::active_request() const noexcept {
  return impl_->active_request;
}

} // namespace studiocast::video
