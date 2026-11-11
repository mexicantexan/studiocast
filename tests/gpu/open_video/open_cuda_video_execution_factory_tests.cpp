#include "core/open_video/video_execution_factory.h"
#include "core/video/video_execution_coordinator.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

namespace {

using namespace studiocast;

std::atomic<bool> g_count_allocations{false};
std::atomic<std::uint64_t> g_allocations{0};

void *Allocate(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_acquire))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!Require((condition), #condition))                                     \
      return false;                                                            \
  } while (false)

hal::ExecutionDomainId Domain(std::uint64_t adapter = 801,
                              std::uint64_t context = 91) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = 0x42;
  domain.logical_context_generation = context;
  domain.allocator_family = hal::AllocatorFamily::cuda_driver;
  domain.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = 0xcafe;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = adapter;
  return domain;
}

hal::ImageDescriptor Image(hal::HostCoherence coherence,
                           hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = 4;
  image.height = 2;
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
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = 12;
  image.planes[0].size_bytes = 24;
  image.planes[0].width = 4;
  image.planes[0].height = 2;
  image.planes[0].alignment_bytes = 1;
  return image;
}

hal::ModelArtifactIdentity Model(std::uint8_t key = 7) {
  hal::ModelArtifactIdentity model;
  model.model_id_hash[0] = key;
  model.artifact_set_hash[0] = static_cast<std::uint8_t>(key + 1);
  model.artifact_generation = 13;
  return model;
}

video::VideoExecutionPlanRequest
Request(hal::SemanticEffectId semantic =
            hal::SemanticEffectId::virtual_background_blur,
        bool required = false) {
  video::VideoExecutionPlanRequest request;
  request.plan_id = 9001;
  request.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  request.intent.backend.backend = hal::BackendId::cuda;
  request.intent.backend.strict = true;
  request.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  request.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::open_cuda;
  request.intent.preferences.strict_effects_engine = true;
  request.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  request.intent.strict_provider_selection = true;
  auto &stage = request.intent.stages[0];
  stage.intent.stage_id = 11;
  stage.intent.semantic = semantic;
  stage.intent.required = required;
  stage.intent.failure_action =
      required ? hal::FailureAction::fail_closed
               : hal::FailureAction::fail_open_preserve_input;
  stage.parameters[0].id = video::CanonicalVideoParameterId::strength;
  stage.parameters[0].kind = video::CanonicalVideoParameterKind::signed_integer;
  stage.parameters[0].signed_value = 8;
  stage.parameter_count = 1;
  stage.model = Model();
  request.intent.stage_count = 1;
  request.host_input =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  request.host_output =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  request.invalidation_key.generations = {21, 22, 23, 24, 25, 91};
  request.invalidation_key.canonical_plan_hash = 0xa1;
  request.invalidation_key.model_artifact_hash = 0xa2;
  request.invalidation_key.execution_domain_hash = 0xa3;
  request.temporal.temporal = false;
  request.temporal.frame_drop_behavior =
      hal::FrameDropBehavior::latest_frame_wins;
  request.budget.persistent_bytes = 1024;
  request.budget.scratch_bytes = 512;
  request.budget.upload_staging_bytes = 24;
  request.budget.readback_staging_bytes = 24;
  request.budget.resource_count = 8;
  request.budget.in_flight_slots = 1;
  request.cpu_tails[0].id = hal::CpuTailId::writer_copy;
  request.cpu_tails[0].after_stage_id = 11;
  request.cpu_tails[0].required = true;
  request.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  request.cpu_tails[0].boundary_index = 1;
  request.cpu_tail_count = 1;
  return request;
}

hal::SemanticCapabilityEvidence
Capability(hal::SemanticEffectId semantic =
               hal::SemanticEffectId::virtual_background_blur) {
  hal::SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  capability.status = hal::StatusId::ready;
  capability.cpu_tail = hal::CpuTailId::writer_copy;
  capability.cpu_tail_placement = hal::DeclaredCpuTail::Placement::after_stage;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  capability.evidence[0] = {
      hal::CapabilityEvidenceCode::provider_runtime_available,
      hal::StatusId::ready,
      hal::ReasonId::none,
      semantic,
      hal::BackendId::cuda,
      true,
  };
  capability.evidence_count = 1;
  return capability;
}

enum class ExecuteMode {
  success,
  optional_fail_open,
  device_loss,
};

class FakeOpenCudaResident final : public hal::IResidentAdapter {
public:
  FakeOpenCudaResident(hal::ExecutionDomainId domain, ExecuteMode mode,
                       std::uint64_t synchronizations)
      : domain_(domain), mode_(mode), synchronizations_(synchronizations) {}

  std::uint64_t adapter_instance_id() const noexcept override {
    return domain_.adapter_instance_id;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::cuda;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    program_ = program;
    candidate->handle_id = 1;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = domain_.adapter_instance_id;
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = domain_;
    candidate->reserved_budget = program.budget;
    readiness->status = hal::StatusId::ready;
    readiness->requested_backend = hal::BackendId::cuda;
    readiness->selected_backend = hal::BackendId::cuda;
    readiness->provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    readiness->execution_domain = domain_;
    readiness->ready_stage_mask = 1;
    readiness->available = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    binding_ = binding;
    tokens_ = {};
    tokens_.input = Resource(101, hal::MemoryDomain::upload_staging, false);
    tokens_.output = Resource(102, hal::MemoryDomain::readback_staging, true);
    *tokens = tokens_;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &program,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    hal::ExecuteResult result;
    result.status = hal::StatusId::executed;
    result.completion = {next_token_++, program.program_id,
                         domain_.adapter_instance_id, request.capture_sequence,
                         true};
    result.output = tokens_.output;
    result.output_valid = true;
    result.stage_count = 1;
    result.stages[0].stage_id = program_.stages[0].stage_id;
    result.stages[0].semantic = program_.stages[0].semantic;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = synchronizations_;
    result.transfers.cpu_tail_boundaries = 1;
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data, 24);
    if (mode_ == ExecuteMode::optional_fail_open) {
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[0].status = hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason = hal::ReasonId::optional_stage_failed_open;
    } else if (mode_ == ExecuteMode::device_loss) {
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_valid = false;
      result.stages[0].status = hal::StageExecutionStatus::required_failed;
      result.stages[0].reason = hal::ReasonId::device_lost;
      result.stages[0].valid_output_preserved = false;
    }
    return result;
  }

  hal::StatusId ResolveHostFrameOutput(
      const hal::PreparedProgramHandle &, const hal::CompletionToken &,
      const hal::ResourceHandle &, hal::HostFrameOutputResolution *resolution,
      hal::ReasonId *reason) noexcept override {
    resolution->descriptor = binding_.output_descriptor;
    resolution->planes = binding_.output_planes;
    resolution->plane_count = binding_.output_plane_count;
    resolution->capture_sequence = binding_.capture_sequence;
    resolution->output_ready = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    return hal::StatusId::ready;
  }

  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
    ++destroy_calls;
    *program = {};
  }
  void Shutdown() noexcept override { ++shutdown_calls; }

  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;

private:
  hal::ResourceHandle Resource(std::uint64_t id, hal::MemoryDomain memory,
                               bool writable) const noexcept {
    hal::ResourceHandle resource;
    resource.resource_id = id;
    resource.allocator_owner_id = domain_.adapter_instance_id;
    resource.lifetime_generation = domain_.logical_context_generation;
    resource.memory_domain = memory;
    resource.lifetime = hal::ResourceLifetime::frame_borrowed;
    resource.synchronization_state = hal::SynchronizationState::ready;
    resource.execution_domain = domain_;
    resource.size_bytes = 24;
    resource.alignment_bytes = 1;
    resource.mutable_resource = writable;
    return resource;
  }

  hal::ExecutionDomainId domain_{};
  ExecuteMode mode_ = ExecuteMode::success;
  std::uint64_t synchronizations_ = 1;
  hal::PreparedProgram program_{};
  hal::HostFrameBinding binding_{};
  hal::HostFrameTokens tokens_{};
  std::uint64_t next_token_ = 1;
};

struct CandidateOwner {
  CandidateOwner(hal::ExecutionDomainId domain, ExecuteMode mode,
                 std::uint64_t synchronizations)
      : adapter(domain, mode, synchronizations) {}
  FakeOpenCudaResident adapter;
};

struct SetupContext {
  hal::ExecutionDomainId domain = Domain();
  ExecuteMode mode = ExecuteMode::success;
  std::uint64_t synchronizations = 1;
  bool fail_prepare = false;
  bool foreign_domain = false;
  bool omit_owner = false;
  std::uint64_t prepare_calls = 0;
  std::uint64_t destructions = 0;
  CandidateOwner *latest_owner = nullptr;
};

SetupContext *g_destroy_context = nullptr;

void DestroyCandidate(void *owner) noexcept {
  delete static_cast<CandidateOwner *>(owner);
  if (g_destroy_context)
    ++g_destroy_context->destructions;
}

hal::AdapterRegistration Registration(FakeOpenCudaResident *adapter,
                                      const hal::ExecutionDomainId &domain,
                                      hal::SemanticEffectId semantic) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = domain.adapter_instance_id;
  registration.backend = hal::BackendId::cuda;
  registration.execution_domain = domain;
  registration.selection_priority = 5;
  registration.resident_adapter = adapter;
  registration.semantic_capabilities[0] = Capability(semantic);
  registration.semantic_capability_count = 1;
  return registration;
}

hal::StatusId
PrepareCandidate(void *opaque, const video::FrozenVideoExecutionPlan &plan,
                 open_cuda::OpenCudaVideoExecutionCandidate *candidate,
                 video::VideoAdapterPrepareEvidence *evidence) noexcept {
  auto *context = static_cast<SetupContext *>(opaque);
  ++context->prepare_calls;
  if (context->fail_prepare) {
    evidence->status = hal::StatusId::unavailable;
    evidence->reason = hal::ReasonId::semantic_provider_unavailable;
    evidence->candidate_resources_rolled_back = true;
    return evidence->status;
  }
  if (context->omit_owner) {
    evidence->status = hal::StatusId::ready;
    evidence->candidate_resources_rolled_back = false;
    return evidence->status;
  }
  auto *owner = new (std::nothrow)
      CandidateOwner(context->domain, context->mode, context->synchronizations);
  if (!owner) {
    evidence->status = hal::StatusId::resource_exhausted;
    evidence->reason = hal::ReasonId::resource_exhausted;
    evidence->candidate_resources_rolled_back = true;
    return evidence->status;
  }
  context->latest_owner = owner;
  g_destroy_context = context;
  candidate->owner = owner;
  candidate->destroy_owner = DestroyCandidate;
  auto registration_domain = context->domain;
  if (context->foreign_domain)
    ++registration_domain.adapter_instance_id;
  candidate->registration =
      Registration(&owner->adapter, registration_domain,
                   plan.prepared_program.stages[0].semantic);
  evidence->status = hal::StatusId::ready;
  evidence->reason = hal::ReasonId::none;
  return evidence->status;
}

hal::ReadinessSnapshot Readiness(const hal::ExecutionDomainId &domain) {
  hal::ReadinessSnapshot readiness;
  readiness.status = hal::StatusId::ready;
  readiness.requested_backend = hal::BackendId::cuda;
  readiness.selected_backend = hal::BackendId::cuda;
  readiness.provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  readiness.execution_domain = domain;
  readiness.ready_stage_mask = 1;
  readiness.available = true;
  return readiness;
}

open_cuda::OpenCudaVideoExecutionFactoryConfig
FactoryConfig(SetupContext *context,
              const video::VideoExecutionPlanRequest &request) {
  open_cuda::OpenCudaVideoExecutionFactoryConfig config;
  config.factory_id = 7001;
  config.selection_priority = 5;
  config.cached.execution_domain = context->domain;
  config.cached.readiness = Readiness(context->domain);
  config.cached.models[0] = request.intent.stages[0].model;
  config.cached.stage_count = 1;
  config.cached.generations = request.invalidation_key.generations;
  config.cached.model_artifact_validated = true;
  config.cached.provider_runtime_validated = true;
  config.cached.device_context_validated = true;
  config.cached.cuda_iobinding_validated = true;
  config.cached.user_stream_bound = true;
  config.cached.reusable_resources_supported = true;
  config.cached.device_preprocess_supported = true;
  config.cached.device_postprocess_supported = true;
  config.cached.one_upload_one_readback = true;
  config.cached.no_intermediate_crossings = true;
  config.cached.stop_aware_bounded_wait = true;
  config.cached.allocation_free_execute = true;
  config.cached.minimum_synchronizations_per_frame = context->synchronizations;
  config.cached.maximum_synchronizations_per_frame = context->synchronizations;
  config.prepare_candidate = PrepareCandidate;
  config.setup_context = context;
  return config;
}

bool BuildFrozen(const video::VideoExecutionPlanRequest &request,
                 const hal::AdapterRegistration &registration,
                 open_cuda::OpenCudaVideoExecutionFactory *factory,
                 video::FrozenVideoExecutionPlan *plan,
                 video::VideoAdapterFactoryPreflight *preflight) {
  hal::StaticAdapterRegistry registry;
  hal::ReasonId reason = hal::ReasonId::none;
  REQUIRE(registry.Register(registration, &reason) == hal::StatusId::ready);
  const auto resolution = hal::ResolveSemanticStages(
      registry, video::BuildVideoResolutionRequest(request.intent));
  REQUIRE(resolution.status == hal::StatusId::ready ||
          resolution.status == hal::StatusId::degraded);
  const auto domain_plan = hal::PlanExecutionDomains(
      registry, resolution, hal::CrossAdapterPolicy::reject);
  REQUIRE(domain_plan.status == hal::StatusId::ready ||
          domain_plan.status == hal::StatusId::degraded);
  *preflight =
      factory->Preflight(request, resolution, domain_plan,
                         video::VideoExecutionRoute::prepared_resident);
  if (preflight->status != hal::StatusId::ready)
    return true;
  plan->plan_id = request.plan_id;
  plan->route = video::VideoExecutionRoute::prepared_resident;
  plan->intent = request.intent;
  plan->host_input = request.host_input;
  plan->host_output = request.host_output;
  plan->invalidation_key = request.invalidation_key;
  plan->temporal = request.temporal;
  plan->budget = request.budget;
  plan->cpu_tails = request.cpu_tails;
  plan->cpu_tail_count = request.cpu_tail_count;
  plan->setup_failure_policy = request.setup_failure_policy;
  plan->migration_policy = request.migration_policy;
  plan->resolution = resolution;
  plan->domain_plan = domain_plan;
  plan->factory_id = factory->Registration().factory_id;
  plan->per_frame_transfers = preflight->per_frame_transfers;
  plan->prepared_program = preflight->prepared_program;
  return true;
}

hal::HostFrameBinding Binding(std::array<std::uint8_t, 24> *input,
                              std::array<std::uint8_t, 24> *output,
                              std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  binding.output_descriptor =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input->data());
  binding.input_planes[0].span_bytes = input->size();
  binding.input_planes[0].row_stride_bytes = 12;
  binding.output_planes[0].write_data =
      reinterpret_cast<std::byte *>(output->data());
  binding.output_planes[0].span_bytes = output->size();
  binding.output_planes[0].row_stride_bytes = 12;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  return binding;
}

bool FullCanonicalCombinationAndOptionalTrackerFallback() {
  SetupContext context;
  auto request = Request();
  const std::array semantics{hal::SemanticEffectId::video_noise_removal,
      hal::SemanticEffectId::eye_contact,
      hal::SemanticEffectId::virtual_background_blur,
      hal::SemanticEffectId::virtual_key_light,
      hal::SemanticEffectId::auto_frame,
      hal::SemanticEffectId::vignette, hal::SemanticEffectId::mirror};
  request.intent.stage_count = semantics.size();
  request.intent.requested_provider = hal::SemanticEffectProviderId::invalid;
  request.intent.strict_provider_selection = false;
  request.temporal = {true, true, hal::FrameDropBehavior::reset_temporal_state,
                      hal::TemporalDiscontinuityAction::reset_and_continue, 5};
  FakeOpenCudaResident adapter(context.domain, ExecuteMode::success, 1);
  auto registration = Registration(&adapter, context.domain, semantics[0]);
  registration.semantic_capability_count = semantics.size();
  auto config = FactoryConfig(&context, request);
  config.cached.stage_count = semantics.size();
  config.cached.readiness.provider = hal::SemanticEffectProviderId::invalid;
  config.cached.readiness.ready_stage_mask = 127;
  config.cached.no_intermediate_crossings = false;
  config.cached.maximum_metadata_readbacks_per_frame = 13;
  config.cached.maximum_metadata_cpu_tails_per_frame = 3;
  config.cached.maximum_metadata_synchronizations_per_frame = 2;
  for (std::size_t index = 0; index < semantics.size(); ++index) {
    auto &stage = request.intent.stages[index];
    stage = request.intent.stages[0];
    stage.intent.stage_id = static_cast<std::uint32_t>(11 + index);
    stage.intent.semantic = semantics[index];
    const bool native = index >= 5;
    stage.model = native ? hal::ModelArtifactIdentity{}
                         : Model(static_cast<std::uint8_t>(index + 7));
    if (index == 3)
      stage.model = request.intent.stages[2].model;
    config.cached.models[index] = stage.model;
    auto &capability = registration.semantic_capabilities[index];
    capability = Capability(semantics[index]);
    capability.provider =
        native ? hal::SemanticEffectProviderId::open_cuda_production_kernels
               : hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    capability.cpu_tail = index + 1 == semantics.size()
                              ? hal::CpuTailId::writer_copy
                              : hal::CpuTailId::invalid;
  }
  request.cpu_tails[0].after_stage_id = 17;
  request.cpu_tails[0].boundary_index = 7;
  open_cuda::OpenCudaVideoExecutionFactory factory(config);
  video::FrozenVideoExecutionPlan plan;
  video::VideoAdapterFactoryPreflight preflight;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(preflight.prepared_program.stage_count == 7);
  REQUIRE(preflight.prepared_program.maximum_metadata_readbacks_per_frame ==
          13);
  REQUIRE(preflight.per_frame_transfers.variable_boundaries_declared);
  REQUIRE(preflight.per_frame_transfers.maximum_intermediate_readbacks == 13);
  REQUIRE(preflight.per_frame_transfers.maximum_cpu_tail_boundaries == 4);
  REQUIRE(preflight.per_frame_transfers.maximum_synchronizations == 3);
  REQUIRE(preflight.prepared_program.requested_provider ==
          hal::SemanticEffectProviderId::invalid);
  for (std::size_t index = 0; index < semantics.size(); ++index)
    REQUIRE(preflight.prepared_program.stages[index].semantic ==
            semantics[index]);
  std::swap(request.intent.stages[0], request.intent.stages[1]);
  request.intent.stages[0].intent.stage_id = 11;
  request.intent.stages[1].intent.stage_id = 12;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);

  // A failed optional tracker preserves the native vignette at fixed center.
  auto fallback = Request(hal::SemanticEffectId::auto_frame);
  fallback.intent.requested_provider = hal::SemanticEffectProviderId::invalid;
  fallback.intent.strict_provider_selection = false;
  fallback.intent.stage_count = 2;
  fallback.intent.stages[1] = fallback.intent.stages[0];
  fallback.intent.stages[1].intent.stage_id = 22;
  fallback.intent.stages[1].intent.semantic = hal::SemanticEffectId::vignette;
  fallback.intent.stages[1].intent.requires_tracked_vignette_center = true;
  fallback.intent.stages[1].model = {};
  fallback.temporal = {true, true, hal::FrameDropBehavior::reset_temporal_state,
                       hal::TemporalDiscontinuityAction::reset_and_continue, 1};
  fallback.cpu_tails[0].after_stage_id = 22;
  fallback.cpu_tails[0].boundary_index = 2;
  auto fallback_config = FactoryConfig(&context, fallback);
  fallback_config.cached.models[0] = {};
  fallback_config.cached.readiness.provider =
      hal::SemanticEffectProviderId::open_cuda_production_kernels;
  open_cuda::OpenCudaVideoExecutionFactory fallback_factory(fallback_config);
  auto fallback_registration =
      Registration(&adapter, context.domain, hal::SemanticEffectId::auto_frame);
  auto &missing = fallback_registration.semantic_capabilities[0];
  missing.status = hal::StatusId::blocked;
  missing.reason = hal::ReasonId::capability_evidence_incomplete;
  missing.production_eligible = missing.exact_evidence = false;
  auto &vignette = fallback_registration.semantic_capabilities[1];
  vignette = Capability(hal::SemanticEffectId::vignette);
  vignette.provider =
      hal::SemanticEffectProviderId::open_cuda_production_kernels;
  vignette.supports_tracked_vignette_center = true;
  fallback_registration.semantic_capability_count = 2;
  REQUIRE(BuildFrozen(fallback, fallback_registration, &fallback_factory, &plan,
                      &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(preflight.prepared_program.stage_count == 1);
  REQUIRE(preflight.prepared_program.stages[0].semantic ==
          hal::SemanticEffectId::vignette);
  REQUIRE(preflight.prepared_program.maximum_metadata_readbacks_per_frame == 0);
  hal::StaticAdapterRegistry fallback_registry;
  hal::ReasonId registration_reason = hal::ReasonId::none;
  REQUIRE(
      fallback_registry.Register(fallback_registration, &registration_reason) ==
      hal::StatusId::ready);
  auto required_tracker = fallback;
  required_tracker.intent.stages[0].intent.required = true;
  required_tracker.intent.stages[0].intent.failure_action =
      hal::FailureAction::fail_closed;
  const auto required_resolution = hal::ResolveSemanticStages(
      fallback_registry,
      video::BuildVideoResolutionRequest(required_tracker.intent));
  REQUIRE(required_resolution.status == hal::StatusId::blocked);
  auto no_tracker = fallback;
  no_tracker.intent.stages[0] = no_tracker.intent.stages[1];
  no_tracker.intent.stage_count = 1;
  no_tracker.temporal = {};
  no_tracker.cpu_tails[0].boundary_index = 1;
  REQUIRE(BuildFrozen(no_tracker, fallback_registration, &fallback_factory,
                      &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);
  return true;
}

bool ExactPreflightAndMalformedRejection() {
  auto request = Request();
  SetupContext context;
  FakeOpenCudaResident registry_adapter(context.domain, ExecuteMode::success,
                                        1);
  auto registration = Registration(&registry_adapter, context.domain,
                                   request.intent.stages[0].intent.semantic);
  open_cuda::OpenCudaVideoExecutionFactory factory(
      FactoryConfig(&context, request));
  video::FrozenVideoExecutionPlan plan;
  video::VideoAdapterFactoryPreflight preflight;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(preflight.cached_evidence_only);
  REQUIRE(preflight.accepts_frozen_resolution);
  REQUIRE(!preflight.performs_selection);
  REQUIRE(preflight.backend == hal::BackendId::cuda);
  REQUIRE(preflight.provider ==
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  REQUIRE(preflight.execution_domain == context.domain);
  REQUIRE(preflight.prepared_program.program_id == request.plan_id);
  REQUIRE(preflight.prepared_program.invalidation_key ==
          request.invalidation_key);
  REQUIRE(preflight.prepared_program.stages[0].model.model_id_hash ==
          request.intent.stages[0].model.model_id_hash);
  REQUIRE(preflight.prepared_program.stages[0].model.artifact_set_hash ==
          request.intent.stages[0].model.artifact_set_hash);
  REQUIRE(preflight.prepared_program.stages[0].model.artifact_generation ==
          request.intent.stages[0].model.artifact_generation);
  REQUIRE(preflight.prepared_program.stages[0].stage_id == 11);
  REQUIRE(preflight.readiness.ready_stage_mask == 1);
  REQUIRE(preflight.prepared_program.cpu_tails[0].id ==
          hal::CpuTailId::writer_copy);
  REQUIRE(preflight.per_frame_transfers.minimum_synchronizations == 1);
  REQUIRE(preflight.per_frame_transfers.maximum_synchronizations == 1);

  auto automatic = request;
  automatic.intent.backend.mode = hal::BackendRequestMode::automatic;
  automatic.intent.backend.backend = hal::BackendId::invalid;
  automatic.intent.backend.strict = false;
  automatic.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::any;
  automatic.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  automatic.intent.preferences.strict_effects_engine = false;
  automatic.intent.requested_provider = hal::SemanticEffectProviderId::invalid;
  automatic.intent.strict_provider_selection = false;
  REQUIRE(BuildFrozen(automatic, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(plan.intent.backend.mode == hal::BackendRequestMode::automatic);
  REQUIRE(plan.intent.backend.backend == hal::BackendId::invalid);
  REQUIRE(preflight.prepared_program.requested_backend == hal::BackendId::cuda);
  REQUIRE(!preflight.prepared_program.strict_backend_selection);
  REQUIRE(preflight.prepared_program.requested_provider ==
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  REQUIRE(!preflight.prepared_program.strict_provider_selection);
  video::VideoAdapterPrepareEvidence automatic_evidence;
  auto automatic_executor = factory.Prepare(plan, &automatic_evidence);
  REQUIRE(automatic_executor != nullptr);
  std::array<std::uint8_t, 24> automatic_input{};
  std::array<std::uint8_t, 24> automatic_output{};
  automatic_input[0] = 0x5a;
  auto automatic_frame = automatic_executor->Execute(
      Binding(&automatic_input, &automatic_output, 77));
  REQUIRE(automatic_frame.status == hal::StatusId::executed);
  REQUIRE(automatic_frame.output_ready);
  hal::ReasonId automatic_reason = hal::ReasonId::none;
  REQUIRE(automatic_executor->CommitOutput(automatic_frame.completion, false,
                                           &automatic_reason) ==
          hal::StatusId::failed);
  REQUIRE(!automatic_executor->Snapshot().latest_committed_frame.valid);
  automatic_frame = automatic_executor->Execute(
      Binding(&automatic_input, &automatic_output, 78));
  REQUIRE(automatic_executor->CommitOutput(automatic_frame.completion, true,
                                           &automatic_reason) ==
          hal::StatusId::ready);
  REQUIRE(
      automatic_executor->Snapshot().latest_committed_frame.capture_sequence ==
      78);

  auto forced_cuda = automatic;
  forced_cuda.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  forced_cuda.intent.backend.backend = hal::BackendId::cuda;
  forced_cuda.intent.backend.strict = true;
  forced_cuda.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  REQUIRE(BuildFrozen(forced_cuda, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(preflight.prepared_program.strict_backend_selection);
  REQUIRE(!preflight.prepared_program.strict_provider_selection);

  auto malformed = request;
  malformed.intent.backend.strict = false;
  REQUIRE(BuildFrozen(malformed, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);

  malformed = request;
  ++malformed.invalidation_key.generations.model;
  REQUIRE(BuildFrozen(malformed, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);
  REQUIRE(preflight.reason == hal::ReasonId::capability_evidence_incomplete);

  malformed = request;
  malformed.intent.stages[0].model = Model(33);
  REQUIRE(BuildFrozen(malformed, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);

  malformed = request;
  malformed.temporal.temporal = true;
  malformed.temporal.history_frames = 1;
  REQUIRE(BuildFrozen(malformed, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);

  malformed = request;
  malformed.host_output.width = 8;
  malformed.host_output.planes[0].width = 8;
  malformed.host_output.planes[0].row_stride_bytes = 24;
  malformed.host_output.planes[0].size_bytes = 48;
  malformed.budget.readback_staging_bytes = 48;
  REQUIRE(BuildFrozen(malformed, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(preflight.prepared_program.host_output.width == 8);
  REQUIRE(preflight.prepared_program.host_output.height == 2);
  REQUIRE(preflight.prepared_program.stages[0].output.image.width == 8);
  REQUIRE(preflight.prepared_program.stages[0].output.image.height == 2);

  auto malformed_config = FactoryConfig(&context, request);
  malformed_config.cached.minimum_synchronizations_per_frame = 2;
  malformed_config.cached.maximum_synchronizations_per_frame = 1;
  open_cuda::OpenCudaVideoExecutionFactory malformed_factory(
      std::move(malformed_config));
  REQUIRE(BuildFrozen(request, registration, &malformed_factory, &plan,
                      &preflight));
  REQUIRE(preflight.status == hal::StatusId::blocked);
  REQUIRE(preflight.reason == hal::ReasonId::capability_evidence_incomplete);
  return true;
}

bool ExecuteTransferCommitAndAllocationContract() {
  auto request = Request();
  SetupContext context;
  // Two synchronization points are intentionally reported by the adapter.
  // The factory must preserve actual counters rather than normalize them to
  // the domain planner's current estimate.
  context.synchronizations = 2;
  FakeOpenCudaResident registry_adapter(context.domain, ExecuteMode::success,
                                        1);
  auto registration = Registration(&registry_adapter, context.domain,
                                   request.intent.stages[0].intent.semantic);
  open_cuda::OpenCudaVideoExecutionFactory factory(
      FactoryConfig(&context, request));
  video::FrozenVideoExecutionPlan plan;
  video::VideoAdapterFactoryPreflight preflight;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  REQUIRE(preflight.status == hal::StatusId::ready);
  REQUIRE(preflight.per_frame_transfers.minimum_synchronizations == 2);
  REQUIRE(preflight.per_frame_transfers.maximum_synchronizations == 2);
  video::VideoAdapterPrepareEvidence evidence;
  auto executor = factory.Prepare(plan, &evidence);
  REQUIRE(executor != nullptr);
  REQUIRE(evidence.status == hal::StatusId::ready);
  REQUIRE(context.prepare_calls == 1);

  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  input.fill(0x5a);
  auto binding = Binding(&input, &output, 1);
  auto frame = executor->Execute(binding);
  REQUIRE(frame.status == hal::StatusId::executed);
  REQUIRE(frame.output_ready);
  REQUIRE(frame.execution_domain == context.domain);
  REQUIRE(frame.transfers.uploads == 1);
  REQUIRE(frame.transfers.final_readbacks == 1);
  REQUIRE(frame.transfers.intermediate_uploads == 0);
  REQUIRE(frame.transfers.intermediate_readbacks == 0);
  REQUIRE(frame.transfers.synchronizations == 2);
  REQUIRE(frame.transfers.cpu_tail_boundaries == 1);
  REQUIRE(output == input);
  REQUIRE(!executor->Snapshot().latest_committed_frame.valid);

  hal::ReasonId reason = hal::ReasonId::none;
  REQUIRE(executor->CommitOutput(frame.completion, false, &reason) ==
          hal::StatusId::failed);
  REQUIRE(reason == hal::ReasonId::output_write_failed);
  REQUIRE(!executor->Snapshot().latest_committed_frame.valid);

  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_release);
  bool succeeded = true;
  for (std::uint64_t sequence = 2; sequence != 66; ++sequence) {
    binding.capture_sequence = sequence;
    const auto next = executor->Execute(binding);
    if (!next.output_ready ||
        executor->CommitOutput(next.completion, true, &reason) !=
            hal::StatusId::ready) {
      succeeded = false;
      break;
    }
  }
  g_count_allocations.store(false, std::memory_order_release);
  REQUIRE(succeeded);
  REQUIRE(g_allocations.load(std::memory_order_relaxed) == 0);
  const auto snapshot = executor->Snapshot();
  REQUIRE(snapshot.latest_committed_frame.valid);
  REQUIRE(snapshot.latest_committed_frame.capture_sequence == 65);
  REQUIRE(snapshot.telemetry.active_backend_frames == 64);
  REQUIRE(snapshot.telemetry.active_backend == hal::BackendId::cuda);
  REQUIRE(snapshot.telemetry.active_provider ==
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  executor->Shutdown();
  REQUIRE(executor->Snapshot().stopped);
  return true;
}

bool OptionalFailureDeviceLossAndRollback() {
  auto request = Request();
  SetupContext context;
  context.mode = ExecuteMode::optional_fail_open;
  FakeOpenCudaResident registry_adapter(context.domain, ExecuteMode::success,
                                        1);
  auto registration = Registration(&registry_adapter, context.domain,
                                   request.intent.stages[0].intent.semantic);
  open_cuda::OpenCudaVideoExecutionFactory factory(
      FactoryConfig(&context, request));
  video::FrozenVideoExecutionPlan plan;
  video::VideoAdapterFactoryPreflight preflight;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  video::VideoAdapterPrepareEvidence evidence;
  auto executor = factory.Prepare(plan, &evidence);
  REQUIRE(executor != nullptr);
  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  input.fill(0x6b);
  auto frame = executor->Execute(Binding(&input, &output, 1));
  REQUIRE(frame.status == hal::StatusId::degraded);
  REQUIRE(frame.output_ready);
  REQUIRE(frame.stages[0].status ==
          hal::StageExecutionStatus::optional_failed_open);
  REQUIRE(frame.stages[0].valid_output_preserved);
  hal::ReasonId reason = hal::ReasonId::none;
  REQUIRE(executor->CommitOutput(frame.completion, true, &reason) ==
          hal::StatusId::ready);
  REQUIRE(executor->Snapshot().latest_committed_frame.valid);
  REQUIRE(executor->Snapshot().telemetry.active_backend_frames == 0);
  executor.reset();

  context.mode = ExecuteMode::device_loss;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  executor = factory.Prepare(plan, &evidence);
  REQUIRE(executor != nullptr);
  frame = executor->Execute(Binding(&input, &output, 2));
  REQUIRE(frame.reason == hal::ReasonId::device_lost);
  REQUIRE(!frame.output_ready);
  REQUIRE(executor->Snapshot().status == hal::StatusId::quarantined);
  executor->Shutdown();

  context.foreign_domain = true;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  executor = factory.Prepare(plan, &evidence);
  REQUIRE(executor == nullptr);
  REQUIRE(evidence.status == hal::StatusId::blocked);
  REQUIRE(evidence.reason == hal::ReasonId::foreign_adapter);
  REQUIRE(evidence.candidate_resources_rolled_back);

  context.foreign_domain = false;
  context.fail_prepare = true;
  REQUIRE(BuildFrozen(request, registration, &factory, &plan, &preflight));
  executor = factory.Prepare(plan, &evidence);
  REQUIRE(executor == nullptr);
  REQUIRE(evidence.status == hal::StatusId::unavailable);
  REQUIRE(evidence.candidate_resources_rolled_back);
  return true;
}

bool CoordinatorStrictSelectionAndStopAwareShutdown() {
  auto request = Request(hal::SemanticEffectId::virtual_background_remove);
  SetupContext context;
  FakeOpenCudaResident registry_adapter(context.domain, ExecuteMode::success,
                                        1);
  auto registration = Registration(&registry_adapter, context.domain,
                                   request.intent.stages[0].intent.semantic);
  hal::StaticAdapterRegistry registry;
  hal::ReasonId reason = hal::ReasonId::none;
  REQUIRE(registry.Register(registration, &reason) == hal::StatusId::ready);
  open_cuda::OpenCudaVideoExecutionFactory factory(
      FactoryConfig(&context, request));
  video::VideoExecutionCoordinator coordinator(registry);
  const auto factory_registration = factory.Registration();
  REQUIRE(coordinator.RegisterFactory(factory_registration, &reason) ==
          hal::StatusId::ready);
  video::VideoExecutionPrepareEvidence prepare;
  REQUIRE(coordinator.Prepare(request, &prepare, &reason) ==
          hal::StatusId::ready);
  REQUIRE(prepare.selected_route ==
          video::VideoExecutionRoute::prepared_resident);
  REQUIRE(prepare.resolution.selected_backend == hal::BackendId::cuda);
  REQUIRE(prepare.resolution.selected_provider ==
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  input.fill(0x7c);
  const auto frame = coordinator.Execute(Binding(&input, &output, 1));
  REQUIRE(frame.status == hal::StatusId::executed);
  REQUIRE(coordinator.CommitOutput(frame.completion, true, &reason) ==
          hal::StatusId::ready);
  const auto snapshot = coordinator.Snapshot();
  REQUIRE(snapshot.latest_committed_frame.valid);
  REQUIRE(snapshot.telemetry.active_backend == hal::BackendId::cuda);
  coordinator.Shutdown();
  REQUIRE(coordinator.Snapshot().stopped);
  REQUIRE(coordinator.Execute(Binding(&input, &output, 2)).status ==
          hal::StatusId::stopped);

  auto substituted = request;
  substituted.intent.backend.backend = hal::BackendId::vulkan;
  REQUIRE(coordinator.Prepare(substituted, &prepare, &reason) ==
          hal::StatusId::stopped);
  return true;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return Allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return Allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, const std::nothrow_t &) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, const std::nothrow_t &) noexcept {
  std::free(memory);
}
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  if (!FullCanonicalCombinationAndOptionalTrackerFallback() ||
      !ExactPreflightAndMalformedRejection() ||
      !ExecuteTransferCommitAndAllocationContract() ||
      !OptionalFailureDeviceLossAndRollback() ||
      !CoordinatorStrictSelectionAndStopAwareShutdown()) {
    return EXIT_FAILURE;
  }
  std::cout << "Open CUDA video execution factory tests passed\n";
  return EXIT_SUCCESS;
}
