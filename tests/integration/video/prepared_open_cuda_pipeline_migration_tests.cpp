#include "core/video/camera_pipeline.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR "."
#endif

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": "            \
                << #condition << '\n';                                         \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

std::uint64_t Align4096(std::uint64_t value) {
  return ((value + 4095u) / 4096u) * 4096u;
}

void TestOddModelGeometryBudget() {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  hal::ResourceBudget budget{};
  CHECK(video::BuildPreparedOpenCudaPipelineResourceBudget(
      641, 479, 257, 263,
      hal::SemanticEffectId::virtual_background_remove, 8, &budget));

  const std::uint64_t frame_rgb = Align4096(641u * 3u) * 479u;
  const std::uint64_t frame_alpha = Align4096(641u * 4u) * 479u;
  const std::uint64_t model_alpha = Align4096(257u * 4u) * 263u;
  const std::uint64_t model_tensor = Align4096(257u * 12u) * 263u;
  const std::uint64_t expected_persistent =
      frame_rgb * 2u + frame_alpha + model_alpha + model_tensor;
  CHECK(budget.persistent_bytes == expected_persistent);
  CHECK(budget.scratch_bytes == 1);
  CHECK(budget.upload_staging_bytes == 641u * 479u * 3u);
  CHECK(budget.readback_staging_bytes == 641u * 479u * 3u);
  CHECK(budget.resource_count == 17);
  CHECK(budget.in_flight_slots == 1);

  CHECK(video::BuildPreparedOpenCudaPipelineResourceBudget(
      641, 479, 257, 263,
      hal::SemanticEffectId::virtual_background_blur, 64, &budget));
  CHECK(budget.scratch_bytes ==
        frame_alpha * 2u + frame_rgb * 2u);
  CHECK(budget.resource_count == 21);

  CHECK(!video::BuildPreparedOpenCudaPipelineResourceBudget(
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max(), 257, 263,
      hal::SemanticEffectId::virtual_background_blur, 64, &budget));
}

void TestWriterCommitAttributionBoundary() {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  CHECK(video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::succeeded, true, hal::StatusId::ready));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::succeeded, false, hal::StatusId::ready));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::succeeded, true, hal::StatusId::failed));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::optional_failed_open, true,
      hal::StatusId::ready));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::required_failed, true,
      hal::StatusId::ready));

  CHECK(!video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::failed, hal::ReasonId::incomplete_execution));
  CHECK(!video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::degraded, hal::ReasonId::optional_stage_failed_open));
  CHECK(video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::quarantined, hal::ReasonId::device_lost));
  CHECK(video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::stopped, hal::ReasonId::shutdown_requested));
}

studiocast::hal::ExecutionDomainId
Domain(studiocast::hal::BackendId backend, std::uint64_t adapter_id) {
  namespace hal = studiocast::hal;
  hal::ExecutionDomainId domain;
  domain.backend = backend;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1c30;
  domain.physical_device.stable_id[0] =
      static_cast<std::uint8_t>(adapter_id);
  domain.logical_context_generation = 31;
  domain.queue_or_stream_identity = 1000 + adapter_id;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.adapter_instance_id = adapter_id;
  if (backend == hal::BackendId::vulkan) {
    domain.api_family = hal::ApiFamily::vulkan;
    domain.allocator_family = hal::AllocatorFamily::vulkan;
    domain.memory_handle_family =
        hal::MemoryHandleFamily::vulkan_buffer;
    domain.synchronization_protocol =
        hal::SynchronizationProtocol::vulkan_queue_fence;
  } else if (backend == hal::BackendId::maxine) {
    domain.api_family = hal::ApiFamily::maxine_nvcv;
    domain.allocator_family = hal::AllocatorFamily::maxine_nvcv;
    domain.memory_handle_family =
        hal::MemoryHandleFamily::nvcv_image;
    domain.synchronization_protocol =
        hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  } else {
    domain.api_family = hal::ApiFamily::cuda_driver;
    domain.allocator_family = hal::AllocatorFamily::cuda_driver;
    domain.memory_handle_family =
        hal::MemoryHandleFamily::cuda_device_pointer;
    domain.synchronization_protocol =
        hal::SynchronizationProtocol::cuda_stream_ordered;
  }
  return domain;
}

studiocast::hal::ProgramGenerations Generations() {
  studiocast::hal::ProgramGenerations generations;
  generations.configuration = 11;
  generations.geometry = 12;
  generations.model = 13;
  generations.provider = 14;
  generations.device = 15;
  generations.context = 16;
  return generations;
}

studiocast::hal::ProgramInvalidationKey Invalidation(
    const studiocast::hal::ProgramGenerations &generations,
    std::uint64_t domain_hash) {
  studiocast::hal::ProgramInvalidationKey key;
  key.generations = generations;
  key.canonical_plan_hash = 101;
  key.model_artifact_hash = 102;
  key.execution_domain_hash = domain_hash;
  return key;
}

studiocast::hal::PreparedProgram Program(
    studiocast::hal::BackendId backend,
    studiocast::hal::SemanticEffectProviderId provider,
    studiocast::hal::SemanticEffectId effect, std::uint64_t adapter_id,
    std::uint64_t program_id, std::uint32_t stage_id,
    const studiocast::hal::ProgramGenerations &generations) {
  namespace hal = studiocast::hal;
  hal::PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = adapter_id;
  program.execution_domain = Domain(backend, adapter_id);
  program.invalidation_key =
      Invalidation(generations, program.execution_domain.queue_or_stream_identity);
  program.requested_backend = backend;
  program.strict_backend_selection = true;
  program.requested_provider = provider;
  program.strict_provider_selection =
      backend == hal::BackendId::cuda;
  program.stage_count = 1;
  program.stages[0].stage_id = stage_id;
  program.stages[0].semantic = effect;
  program.stages[0].provider = provider;
  program.stages[0].backend = backend;
  program.stages[0].execution_domain = program.execution_domain;
  program.stages[0].failure_action =
      hal::FailureAction::fail_open_preserve_input;
  return program;
}

studiocast::hal::ReadinessSnapshot Readiness(
    const studiocast::hal::PreparedProgram &program) {
  studiocast::hal::ReadinessSnapshot readiness;
  readiness.status = studiocast::hal::StatusId::ready;
  readiness.requested_backend = program.requested_backend;
  readiness.selected_backend = program.execution_domain.backend;
  readiness.provider = program.stages[0].provider;
  readiness.execution_domain = program.execution_domain;
  readiness.ready_stage_mask = 1;
  readiness.available = true;
  return readiness;
}

studiocast::hal::CommittedFrameSnapshot Commit(
    const studiocast::hal::PreparedProgram &program,
    std::uint64_t capture_sequence, std::uint64_t output_sequence,
    studiocast::hal::StageExecutionStatus status =
        studiocast::hal::StageExecutionStatus::succeeded) {
  namespace hal = studiocast::hal;
  hal::CommittedFrameSnapshot committed;
  committed.valid = true;
  committed.program_id = program.program_id;
  committed.invalidation_key = program.invalidation_key;
  committed.execution_domain = program.execution_domain;
  committed.capture_sequence = capture_sequence;
  committed.output_commit_sequence = output_sequence;
  committed.stage_count = 1;
  committed.stages[0].stage_id = program.stages[0].stage_id;
  committed.stages[0].semantic = program.stages[0].semantic;
  committed.stages[0].backend = program.stages[0].backend;
  committed.stages[0].provider = program.stages[0].provider;
  committed.stages[0].status = status;
  committed.stages[0].reason =
      status == hal::StageExecutionStatus::optional_failed_open
          ? hal::ReasonId::optional_stage_failed_open
          : hal::ReasonId::none;
  return committed;
}

void AddVulkanEvidence(studiocast::hal::ResolvedStage *stage) {
  namespace hal = studiocast::hal;
  if (!stage)
    return;
  constexpr std::array common{
      hal::CapabilityEvidenceCode::vulkan_build_enabled,
      hal::CapabilityEvidenceCode::vulkan_loader_available,
      hal::CapabilityEvidenceCode::vulkan_physical_device_available,
      hal::CapabilityEvidenceCode::vulkan_non_cpu_device,
      hal::CapabilityEvidenceCode::vulkan_compute_queue_available,
      hal::CapabilityEvidenceCode::vulkan_context_healthy,
      hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready,
      hal::CapabilityEvidenceCode::vulkan_exact_device_shared,
      hal::CapabilityEvidenceCode::vulkan_device_local_residency,
      hal::CapabilityEvidenceCode::vulkan_bounded_allocations,
      hal::CapabilityEvidenceCode::vulkan_no_cpu_layers,
      hal::CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
  };
  for (const auto code : common) {
    stage->evidence[stage->evidence_count++] = {
        code, hal::StatusId::ready, hal::ReasonId::none,
        stage->intent.semantic, hal::BackendId::vulkan, true};
  }
  stage->evidence[stage->evidence_count++] = {
      stage->intent.semantic == hal::SemanticEffectId::mirror
          ? hal::CapabilityEvidenceCode::open_vulkan_mirror_production_ready
          : hal::CapabilityEvidenceCode::
                open_vulkan_vignette_fixed_center_production_ready,
      hal::StatusId::ready, hal::ReasonId::none, stage->intent.semantic,
      hal::BackendId::vulkan, true};
}

studiocast::video::PreparedHardwareExecutionSnapshot VulkanSnapshot(
    const studiocast::hal::ProgramGenerations &generations,
    std::uint64_t capture_sequence, std::uint64_t output_sequence) {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  const auto program = Program(
      hal::BackendId::vulkan,
      hal::SemanticEffectProviderId::open_vulkan_production_kernels,
      hal::SemanticEffectId::mirror, 101, 1001, 91, generations);
  video::PreparedHardwareExecutionSnapshot snapshot;
  snapshot.route = video::PreparedHardwareRoute::hal_live;
  snapshot.eligibility = video::PreparedHardwareEligibility::eligible;
  snapshot.prepared = true;
  snapshot.active_program.valid = true;
  snapshot.active_program.program_id = program.program_id;
  snapshot.active_program.owner_adapter_instance_id =
      program.owner_adapter_instance_id;
  snapshot.active_program.invalidation_key = program.invalidation_key;
  snapshot.active_program.execution_domain = program.execution_domain;
  snapshot.active_program.stage_count = 1;
  snapshot.readiness = Readiness(program);
  snapshot.telemetry.runtime_status = hal::StatusId::executed;
  snapshot.telemetry.transfers.uploads = 1;
  snapshot.telemetry.transfers.final_readbacks = 1;
  snapshot.telemetry.transfers.synchronizations = 1;
  snapshot.latest_committed_frame =
      Commit(program, capture_sequence, output_sequence);
  snapshot.setup.resolution.request_mode =
      hal::BackendRequestMode::explicit_backend;
  snapshot.setup.resolution.strict_backend_selection = true;
  snapshot.setup.resolution.status = hal::StatusId::ready;
  snapshot.setup.resolution.requested_backend = hal::BackendId::vulkan;
  snapshot.setup.resolution.selected_backend = hal::BackendId::vulkan;
  snapshot.setup.resolution.selected_provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  snapshot.setup.resolution.available = true;
  snapshot.setup.resolution.stage_count = 1;
  auto &resolved = snapshot.setup.resolution.stages[0];
  resolved.intent.stage_id = program.stages[0].stage_id;
  resolved.intent.semantic = program.stages[0].semantic;
  resolved.intent.failure_action =
      hal::FailureAction::fail_open_preserve_input;
  resolved.status = hal::StatusId::ready;
  resolved.selected_backend = hal::BackendId::vulkan;
  resolved.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  resolved.adapter_instance_id = program.owner_adapter_instance_id;
  resolved.execution_domain = program.execution_domain;
  AddVulkanEvidence(&resolved);
  return snapshot;
}

studiocast::video::PreparedMaxineHardwareExecutionSnapshot MaxineSnapshot(
    const studiocast::hal::ProgramGenerations &generations,
    std::uint64_t capture_sequence, std::uint64_t output_sequence) {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  const auto program =
      Program(hal::BackendId::maxine,
              hal::SemanticEffectProviderId::maxine_ar,
              hal::SemanticEffectId::eye_contact, 102, 1002, 92,
              generations);
  video::PreparedMaxineHardwareExecutionSnapshot snapshot;
  snapshot.route = video::PreparedMaxineHardwareRoute::hal_live;
  snapshot.eligibility =
      video::PreparedMaxineHardwareEligibility::eligible;
  snapshot.active_program = program;
  snapshot.execution_domain = program.execution_domain;
  snapshot.readiness = Readiness(program);
  snapshot.telemetry.runtime_status = hal::StatusId::executed;
  snapshot.telemetry.transfers.uploads = 1;
  snapshot.telemetry.transfers.final_readbacks = 1;
  snapshot.telemetry.transfers.synchronizations = 1;
  snapshot.telemetry.transfers.cpu_tail_boundaries = 1;
  snapshot.latest_committed_frame =
      Commit(program, capture_sequence, output_sequence);
  snapshot.prepared = true;
  return snapshot;
}

studiocast::video::PreparedOpenCudaHardwareExecutionSnapshot
OpenCudaSnapshot(const studiocast::hal::ProgramGenerations &generations,
                 std::uint64_t capture_sequence,
                 std::uint64_t output_sequence) {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  const auto program =
      Program(hal::BackendId::cuda,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              hal::SemanticEffectId::virtual_background_blur, 103, 1003, 93,
              generations);
  video::PreparedOpenCudaHardwareExecutionSnapshot snapshot;
  snapshot.route = video::PreparedOpenCudaHardwareRoute::hal_live;
  snapshot.eligibility =
      video::PreparedOpenCudaHardwareEligibility::eligible;
  snapshot.active_program = program;
  snapshot.execution_domain = program.execution_domain;
  snapshot.readiness = Readiness(program);
  snapshot.telemetry.runtime_status = hal::StatusId::executed;
  snapshot.telemetry.transfers.uploads = 1;
  snapshot.telemetry.transfers.final_readbacks = 1;
  snapshot.telemetry.transfers.synchronizations = 1;
  snapshot.telemetry.transfers.cpu_tail_boundaries = 1;
  snapshot.latest_committed_frame =
      Commit(program, capture_sequence, output_sequence);
  snapshot.production.prepared_capability_committed = true;
  snapshot.prepared = true;
  return snapshot;
}

void TestTypedPipelineStatusProjection() {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  const auto generations = Generations();
  video::effects::BroadcastCameraEffects effects;
  video::CameraPipelineHalStatusProjectionInput input;
  input.effects = &effects;
  input.lifecycle = video::VideoRuntimeLifecycle::running;
  input.status_generation = 41;
  input.configuration_generation = generations.configuration;
  input.geometry_generation = generations.geometry;
  video::MultiIslandVideoRuntimeStatusSnapshot snapshot;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  CHECK(snapshot.island_count == 0);
  CHECK(snapshot.effect_count == video::kCanonicalVideoEffectCount);
  CHECK(snapshot.aggregate.transfers.uploads == 0);

  effects.eye_contact.enabled = true;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  const auto &automatic_eye = snapshot.effects[1];
  CHECK(automatic_eye.requested);
  CHECK(automatic_eye.requested_backend == hal::BackendId::maxine);
  CHECK(!automatic_eye.strict_backend_selection);
  CHECK(automatic_eye.blocked);
  CHECK(!automatic_eye.available);
  CHECK(!snapshot.authority.complete);
  CHECK(snapshot.authority.prepared_route_effect_mask == 0);
  CHECK(snapshot.authority.compatibility_uncovered_effect_mask ==
        (std::uint16_t{1} << 1u));

  effects.mirror = true;
  effects.engine = video::effects::EffectsEnginePreference::maxine;
  input.compute_backend = video::ComputeBackendPreference::vulkan;
  input.final_visual_live = true;
  input.maxine_resident_live = true;
  input.final_visual = VulkanSnapshot(generations, 501, 9);
  input.maxine_resident = MaxineSnapshot(generations, 501, 9);
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  CHECK(snapshot.island_count == 2);
  CHECK(snapshot.aggregate.transfers.uploads == 2);
  CHECK(snapshot.aggregate.transfers.final_readbacks == 2);
  CHECK(snapshot.aggregate.transfers.cpu_tail_boundaries == 1);
  CHECK(snapshot.effects[1].active);
  CHECK(snapshot.effects[8].active);
  CHECK(snapshot.effects[8].active_backend == hal::BackendId::vulkan);
  CHECK(snapshot.authority.complete);
  CHECK(snapshot.authority.compatibility_uncovered_effect_mask == 0);

  const auto qualified_maxine = input.maxine_resident;
  input.maxine_resident.eligibility =
      video::PreparedMaxineHardwareEligibility::runtime_unavailable;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::invalid_island);
  CHECK(snapshot.status_generation == 0);

  input.maxine_resident = qualified_maxine;
  input.maxine_resident.latest_committed_frame.output_commit_sequence = 8;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::mixed_writer_commit);
  CHECK(snapshot.status_generation == 0);

  input.maxine_resident_live = false;
  input.final_visual = VulkanSnapshot(generations, 502, 10);
  input.final_visual.setup.resolution.stages[0].intent.semantic =
      hal::SemanticEffectId::video_noise_removal;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::invalid_capability_evidence);
  CHECK(snapshot.status_generation == 0);

  effects = {};
  effects.engine = video::effects::EffectsEnginePreference::open_cuda;
  effects.virtual_background.mode =
      video::effects::VirtualBackgroundMode::blur;
  input = {};
  input.effects = &effects;
  input.compute_backend = video::ComputeBackendPreference::cuda;
  input.lifecycle = video::VideoRuntimeLifecycle::running;
  input.status_generation = 50;
  input.configuration_generation = generations.configuration;
  input.geometry_generation = generations.geometry;
  input.open_cuda_virtual_background_live = true;
  input.open_cuda_virtual_background =
      OpenCudaSnapshot(generations, 601, 10);
  input.open_cuda_virtual_background.latest_committed_frame = {};
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  CHECK(snapshot.island_count == 1);
  CHECK(snapshot.effects[2].selected);
  CHECK(!snapshot.effects[2].active);
  CHECK(!snapshot.writer_commit.valid);

  input.open_cuda_virtual_background.latest_committed_frame =
      OpenCudaSnapshot(generations, 601, 10).latest_committed_frame;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  CHECK(snapshot.effects[2].active);
  CHECK(snapshot.writer_commit.valid);

  input.open_cuda_virtual_background.latest_committed_frame =
      Commit(input.open_cuda_virtual_background.active_program, 602, 11,
             hal::StageExecutionStatus::optional_failed_open);
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  CHECK(snapshot.effects[2].degraded);
  CHECK(snapshot.effects[2].fallback);
  CHECK(!snapshot.effects[2].active);

  auto evidence_free_open_cuda =
      OpenCudaSnapshot(generations, 602, 12);
  evidence_free_open_cuda.production.prepared_capability_committed = false;
  input.open_cuda_virtual_background = evidence_free_open_cuda;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::invalid_island);
  CHECK(snapshot.status_generation == 0);

  input.open_cuda_virtual_background =
      OpenCudaSnapshot(generations, 602, 12);
  effects.engine = video::effects::EffectsEnginePreference::maxine;
  input.status_generation++;
  const auto strict_mismatch =
      video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot);
  if (strict_mismatch.code !=
      video::MultiIslandStatusValidationCode::strict_backend_substitution) {
    std::cerr << "strict mismatch validation code="
              << static_cast<unsigned int>(strict_mismatch.code) << '\n';
  }
  CHECK(strict_mismatch.code ==
        video::MultiIslandStatusValidationCode::strict_backend_substitution);

  input.open_cuda_virtual_background_live = false;
  input.lifecycle = video::VideoRuntimeLifecycle::idle;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::valid);
  CHECK(snapshot.island_count == 0);
  CHECK(!snapshot.writer_commit.valid);
  CHECK(!snapshot.effects[2].active);

  input.lifecycle = video::VideoRuntimeLifecycle::running;
  input.open_cuda_virtual_background_live = true;
  input.open_cuda_virtual_background =
      OpenCudaSnapshot(generations, 603, 12);
  input.open_cuda_virtual_background.active_program.stage_count =
      input.open_cuda_virtual_background.active_program.stages.size() + 1;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::invalid_island);
  CHECK(snapshot.status_generation == 0);

  input.open_cuda_virtual_background =
      OpenCudaSnapshot(generations, 604, 13);
  input.open_cuda_virtual_background.latest_committed_frame.stage_count =
      input.open_cuda_virtual_background.latest_committed_frame.stages.size() +
      1;
  input.status_generation++;
  CHECK(video::BuildCameraPipelineHalStatusSnapshot(input, &snapshot).code ==
        video::MultiIslandStatusValidationCode::invalid_island);
  CHECK(snapshot.status_generation == 0);
}

std::string ReadPipelineSource() {
  const auto root = std::filesystem::path(STUDIOCAST_SOURCE_DIR);
  std::ifstream input(root / "src/core/video/camera_pipeline.cpp");
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

void TestMigrationStructuralGuards() {
  const std::string source = ReadPipelineSource();
  CHECK(!source.empty());
  if (source.empty())
    return;

  const auto setup = source.find(
      "try_prepare_open_cuda_virtual_background");
  const auto loop = source.find("while (!stop_.load())");
  const auto execute = source.find(
      "open_cuda_hardware_composition.Execute(binding)");
  const auto write = source.find("writer_.WriteFrameDetailed");
  const auto commit_gate = source.find(
      "PreparedOpenCudaPipelineMayCommitAttribution", write);
  CHECK(setup != std::string::npos && loop != std::string::npos &&
        setup < loop);
  CHECK(execute != std::string::npos && execute > loop);
  CHECK(write != std::string::npos && commit_gate != std::string::npos &&
        write < commit_gate);

  CHECK(source.find(
            "fx.engine == effects::EffectsEnginePreference::open_cuda") !=
        std::string::npos);
  CHECK(source.find("request.compute = PreparedOpenCudaComputeIntent::cuda") !=
        std::string::npos);
  CHECK(source.find(
            "request.semantic_stage_count = plan.ordered_effect_ids.size()") !=
        std::string::npos);
  CHECK(source.find("request.tensorrt_enabled = "
                    "OpenCudaTensorRtEnabledFromEnv()") !=
        std::string::npos);
  CHECK(source.find("outA.format == PixelFormat::rgb24") !=
        std::string::npos);
  CHECK(source.find("request.scaling_required =") != std::string::npos);
  CHECK(source.find("request.output_format_conversion_required =") !=
        std::string::npos);
  CHECK(source.find("ValidateModelArtifactIdentity") != std::string::npos);
  CHECK(source.find("AcquireProductionSetupLease") !=
        std::string::npos);
  CHECK(source.find(
            "open_cuda_hardware_composition.PrepareWithSetupLease") !=
        std::string::npos);
  CHECK(source.find(
            "open_cuda_hardware_composition.Invalidate") !=
        std::string::npos);
  CHECK(source.find("open_cuda_hardware_composition.Shutdown") !=
        std::string::npos);
  CHECK(source.find("PreparedOpenCudaHardwareExecutionOwner "
                    "prepared_open_cuda_hardware_owner") ==
        std::string::npos);
  CHECK(source.find(
            "prepared_open_cuda_hardware_owner.ExecuteHostFrame") ==
        std::string::npos);
  CHECK(source.find(
            "prepared_open_cuda_hardware_owner.CommitOutput") ==
        std::string::npos);
  CHECK(source.find("Never retry through the legacy context") !=
        std::string::npos);
  CHECK(source.find("breaker.AllowsAttempt(capture_sequence)") !=
        std::string::npos);

  // Provider probing, model scanning, artifact hashing, setup-lease
  // acquisition and JSON must remain before the capture loop. The neutral
  // composition is the sole facade in the live section.
  const std::string live = source.substr(loop);
  CHECK(live.find("ModelPackRegistry::ScanDefault") == std::string::npos);
  CHECK(live.find("ValidateModelArtifactIdentity") == std::string::npos);
  CHECK(live.find("AcquireProductionSetupLease") ==
        std::string::npos);
  CHECK(live.find("PrepareWithSetupLease") ==
        std::string::npos);
  CHECK(live.find("Json") == std::string::npos);
  CHECK(live.find("JSON") == std::string::npos);
}

} // namespace

int main() {
  TestOddModelGeometryBudget();
  TestWriterCommitAttributionBoundary();
  TestTypedPipelineStatusProjection();
  TestMigrationStructuralGuards();
  if (failures != 0) {
    std::cerr << failures
              << " prepared Open CUDA pipeline migration checks failed\n";
    return 1;
  }
  std::cout << "prepared Open CUDA pipeline migration checks passed\n";
  return 0;
}
