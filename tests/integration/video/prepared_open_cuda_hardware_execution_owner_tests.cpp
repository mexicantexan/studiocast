#include "core/video/prepared_open_cuda_hardware_execution_owner.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

namespace {

namespace hal = studiocast::hal;
namespace open_video = studiocast::open_video;
namespace video = studiocast::video;

std::atomic<std::uint64_t> g_allocations{0};
std::atomic<bool> g_count_allocations{false};

void *Allocate(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_acquire))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": "   \
                << #condition << '\n';                                         \
      return false;                                                            \
    }                                                                          \
  } while (false)

hal::ExecutionDomainId Domain(std::uint64_t adapter = 71,
                              std::uint64_t context = 9,
                              std::uint64_t stream = 0xCA71) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = 0x41;
  domain.logical_context_generation = context;
  domain.allocator_family = hal::AllocatorFamily::cuda_driver;
  domain.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = stream;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = adapter;
  return domain;
}

open_video::ValidatedModelArtifactIdentity Artifact(std::uint8_t key = 1) {
  open_video::ValidatedModelArtifactIdentity artifact;
  artifact.registry_generation = 7;
  artifact.model_id_sha256[0] = key;
  artifact.manifest_sha256[0] = static_cast<std::uint8_t>(key + 1);
  artifact.artifact_sha256[0] = static_cast<std::uint8_t>(key + 2);
  artifact.artifact_set_sha256[0] = static_cast<std::uint8_t>(key + 3);
  return artifact;
}

open_video::ModelPack
Pack(const open_video::ValidatedModelArtifactIdentity &artifact) {
  open_video::ModelPack pack;
  pack.id = "hermetic-open-cuda-matting";
  pack.display_name = "Hermetic Open CUDA matting";
  pack.task = "matting";
  pack.registry_generation = artifact.registry_generation;
  pack.registry_manifest_sha256 = artifact.manifest_sha256;
  open_video::ModelFile file;
  file.name = "model.onnx";
  file.kind = "onnx";
  file.role = "main";
  pack.files.push_back(file);
  open_video::MattingSpec matting;
  matting.input.name = "input";
  matting.input.layout = "nchw";
  matting.input.dtype = "float32";
  matting.input.width = 16;
  matting.input.height = 8;
  matting.input.channels = 3;
  matting.output.name = "alpha";
  matting.output.kind = "alpha";
  matting.output.dtype = "float32";
  matting.output.layout = "nchw";
  matting.output.width = 16;
  matting.output.height = 8;
  matting.output.channels = 1;
  matting.preprocess.color = "rgb";
  matting.preprocess.range = "0..1";
  pack.matting = matting;
  return pack;
}

bool StopRequested(void *) noexcept { return false; }
bool CancelRequested(void *) noexcept { return false; }
std::uint64_t NowNanoseconds(void *) noexcept {
  static std::atomic<std::uint64_t> now{0};
  return now.fetch_add(1000, std::memory_order_relaxed);
}
void PollWait(void *, std::uint64_t) noexcept {}

video::PreparedOpenCudaHardwareRequest
Request(hal::SemanticEffectId semantic =
            hal::SemanticEffectId::virtual_background_blur,
        bool required = false) {
  const auto artifact = Artifact();
  video::PreparedOpenCudaHardwareRequest request;
  request.effects_engine =
      video::PreparedOpenCudaEffectsEngineIntent::open_cuda;
  request.compute = video::PreparedOpenCudaComputeIntent::cuda;
  request.requested_backend = hal::BackendId::cuda;
  request.strict_backend_selection = true;
  request.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  request.strict_provider_selection = true;
  request.stage_id = 3;
  request.semantic = semantic;
  request.semantic_stage_count = 1;
  request.required = required;
  request.single_resident_island = true;
  request.capture_width = 16;
  request.capture_height = 8;
  request.output_width = 16;
  request.output_height = 8;
  request.host_input_is_tight_rgb24 = true;
  request.host_output_is_tight_rgb24 = true;
  request.generations = {11, 12, 13, 14, 15, 0};
  request.capacity = {256 * 1024, 128 * 1024, 4096, 4096, 32, 1};
  request.model_artifact = artifact;
  request.model_pack = Pack(artifact);
  request.strength = 8;
  request.remove_background_rgb = {0x10, 0x20, 0x30};
  request.wait.timeout_nanoseconds = 5'000'000;
  request.wait.poll_interval_nanoseconds = 50'000;
  request.wait.maximum_poll_count = 100;
  request.wait.stop_requested = StopRequested;
  request.wait.cancel_requested = CancelRequested;
  request.wait.now_nanoseconds = NowNanoseconds;
  request.wait.poll_wait = PollWait;
  return request;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t id, hal::MemoryDomain memory,
                             std::uint64_t bytes) {
  hal::ResourceHandle resource;
  resource.resource_id = id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = memory;
  resource.lifetime = hal::ResourceLifetime::frame_borrowed;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = bytes;
  resource.alignment_bytes = 1;
  resource.mutable_resource = memory == hal::MemoryDomain::readback_staging;
  return resource;
}

class FakeCudaAdapter final : public hal::IResidentAdapter {
public:
  enum class ExecuteMode {
    success,
    optional_fail_open,
    required_failure,
    device_loss,
    foreign_output,
    wrong_generation,
  };

  std::uint64_t adapter_instance_id() const noexcept override { return 71; }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::cuda;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    last_prepared = program;
    if (fail_prepare) {
      readiness->status = hal::StatusId::resource_exhausted;
      readiness->reason = hal::ReasonId::resource_exhausted;
      if (reason)
        *reason = hal::ReasonId::resource_exhausted;
      return hal::StatusId::resource_exhausted;
    }
    program_ = program;
    candidate->handle_id = next_handle_++;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = adapter_instance_id();
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    if (oversized_budget)
      ++candidate->reserved_budget.persistent_bytes;
    readiness->status = hal::StatusId::degraded;
    readiness->reason = hal::ReasonId::cpu_tail_required;
    readiness->requested_backend = hal::BackendId::cuda;
    readiness->selected_backend =
        substitute_backend ? hal::BackendId::vulkan : hal::BackendId::cuda;
    readiness->provider =
        substitute_provider
            ? hal::SemanticEffectProviderId::open_video_onnxruntime_cpu
            : hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = 1;
    readiness->available = true;
    readiness->degraded = true;
    if (reason)
      *reason = hal::ReasonId::cpu_tail_required;
    return hal::StatusId::degraded;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    ++bind_calls;
    binding_ = binding;
    tokens->input = Resource(program_, 1000 + binding.capture_sequence,
                             hal::MemoryDomain::upload_staging,
                             binding.input_planes[0].span_bytes);
    tokens->output = Resource(program_, 2000 + binding.capture_sequence,
                              hal::MemoryDomain::readback_staging,
                              binding.output_planes[0].span_bytes);
    tokens_ = *tokens;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    hal::ExecuteResult result;
    result.status = hal::StatusId::executed;
    result.completion.token_id = next_token_++;
    result.completion.program_id = program_.program_id;
    result.completion.adapter_instance_id = adapter_instance_id();
    result.completion.capture_sequence = request.capture_sequence;
    result.completion.complete = true;
    result.output = tokens_.output;
    result.output_valid = true;
    result.stage_count = 1;
    result.stages[0].stage_id = program_.stages[0].stage_id;
    result.stages[0].semantic = program_.stages[0].semantic;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    result.transfers.cpu_tail_boundaries = 1;
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data,
                static_cast<std::size_t>(binding_.output_planes[0].span_bytes));
    switch (mode) {
    case ExecuteMode::success:
      break;
    case ExecuteMode::optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[0].status = hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason = hal::ReasonId::optional_stage_failed_open;
      break;
    case ExecuteMode::required_failure:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::required_stage_failed;
      result.output_valid = false;
      result.stages[0].status = hal::StageExecutionStatus::required_failed;
      result.stages[0].reason = hal::ReasonId::required_stage_failed;
      break;
    case ExecuteMode::device_loss:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_valid = false;
      break;
    case ExecuteMode::foreign_output:
      ++result.output.execution_domain.adapter_instance_id;
      break;
    case ExecuteMode::wrong_generation:
      ++result.output.lifetime_generation;
      break;
    }
    return result;
  }

  hal::StatusId ResolveHostFrameOutput(
      const hal::PreparedProgramHandle &, const hal::CompletionToken &,
      const hal::ResourceHandle &, hal::HostFrameOutputResolution *resolution,
      hal::ReasonId *reason) noexcept override {
    ++resolve_calls;
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

  bool fail_prepare = false;
  bool oversized_budget = false;
  bool substitute_provider = false;
  bool substitute_backend = false;
  ExecuteMode mode = ExecuteMode::success;
  std::uint64_t prepare_calls = 0;
  std::uint64_t bind_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t resolve_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;
  hal::PreparedProgram last_prepared{};

private:
  std::uint64_t next_handle_ = 1;
  std::uint64_t next_token_ = 1;
  hal::PreparedProgram program_{};
  hal::HostFrameBinding binding_{};
  hal::HostFrameTokens tokens_{};
};

hal::AdapterRegistration Registration(FakeCudaAdapter *adapter) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = adapter->adapter_instance_id();
  registration.backend = hal::BackendId::cuda;
  registration.execution_domain = Domain();
  registration.selection_priority = 10;
  registration.resident_adapter = adapter;
  for (const auto semantic :
       {hal::SemanticEffectId::virtual_background_blur,
        hal::SemanticEffectId::virtual_background_remove}) {
    auto &capability =
        registration
            .semantic_capabilities[registration.semantic_capability_count++];
    capability.semantic = semantic;
    capability.provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    capability.status = hal::StatusId::ready;
    capability.reason = hal::ReasonId::none;
    capability.live_stage = true;
    capability.production_eligible = true;
    capability.exact_evidence = true;
  }
  return registration;
}

struct Frames {
  std::array<std::uint8_t, 16 * 8 * 3> input{};
  std::array<std::uint8_t, 16 * 8 * 3> output{};
};

bool EligibilityMatrixFailsClosed() {
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(Request()) ==
        video::PreparedOpenCudaHardwareEligibility::eligible);
  auto request = Request();
  request.effects_engine =
      video::PreparedOpenCudaEffectsEngineIntent::automatic;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::
            automatic_engine_compatibility);
  request = Request();
  request.effects_engine = video::PreparedOpenCudaEffectsEngineIntent::maxine;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::not_explicit_open_cuda);
  request = Request();
  request.compute = video::PreparedOpenCudaComputeIntent::vulkan;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::compute_family_disallowed);
  request = Request();
  request.strict_backend_selection = false;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::not_strict_cuda);
  request = Request();
  request.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::provider_mismatch);
  request = Request();
  request.tensorrt_enabled = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::tensorrt_unsupported);
  request = Request();
  request.semantic_stage_count = 2;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::stage_count_invalid);
  request = Request(hal::SemanticEffectId::virtual_background_replace);
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::semantic_unsupported);
  request = Request();
  request.contains_other_resident_stage = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::mixed_resident_section);
  request = Request();
  request.contains_other_execution_domain = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::cross_domain_section);
  request = Request();
  request.request_vignette = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::unsupported_effect);
  request = Request();
  request.shared_matte_required = true;
  CHECK(
      video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
      video::PreparedOpenCudaHardwareEligibility::shared_analysis_unsupported);
  request = Request();
  request.temporal_history_required = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::temporal_stage_unsupported);
  request = Request();
  request.host_output_is_tight_rgb24 = false;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::unsupported_host_layout);
  request = Request();
  request.output_width = 8;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::geometry_mismatch);
  request = Request();
  request.scaling_required = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::scaling_unsupported);
  request = Request();
  request.output_format_conversion_required = true;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::
            output_conversion_unsupported);
  request = Request();
  request.generations.provider = 0;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::invalid_generation);
  request = Request();
  request.model_artifact = {};
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::
            model_artifact_evidence_incomplete);
  request = Request();
  request.model_pack.registry_generation++;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::model_pack_incompatible);
  request = Request();
  request.capacity.resource_count = 0;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::resource_budget_invalid);
  request = Request();
  request.wait.stop_requested = nullptr;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::wait_policy_invalid);
  request = Request();
  request.declares_writer_copy_cpu_tail = false;
  CHECK(video::EvaluatePreparedOpenCudaHardwareEligibility(request) ==
        video::PreparedOpenCudaHardwareEligibility::writer_tail_invalid);
  return true;
}

bool ExactPreparedProgramAndStrictSubstitution() {
  FakeCudaAdapter adapter;
  video::PreparedOpenCudaHardwareExecutionOwner owner(Registration(&adapter));
  auto request = Request();
  const auto setup = owner.Configure(request);
  CHECK(setup.route == video::PreparedOpenCudaHardwareRoute::hal_live);
  CHECK(setup.status == hal::StatusId::degraded);
  const auto snapshot = owner.Snapshot();
  CHECK(snapshot.prepared);
  CHECK(snapshot.active_program.stage_count == 1);
  CHECK(snapshot.active_program.cpu_tail_count == 1);
  CHECK(snapshot.active_program.cpu_tails[0].id == hal::CpuTailId::writer_copy);
  CHECK(snapshot.active_program.execution_domain == Domain());
  CHECK(snapshot.active_program.requested_backend == hal::BackendId::cuda);
  CHECK(snapshot.active_program.strict_backend_selection);
  CHECK(snapshot.active_program.requested_provider ==
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  CHECK(snapshot.active_program.strict_provider_selection);
  CHECK(snapshot.active_program.strict_execution_domain);
  CHECK(snapshot.active_program.stages[0].semantic == request.semantic);
  CHECK(snapshot.active_program.stages[0].failure_action ==
        hal::FailureAction::fail_open_preserve_input);
  CHECK(snapshot.active_program.stages[0].model.model_id_hash ==
        request.model_artifact.model_id_sha256);
  CHECK(snapshot.active_program.stages[0].model.artifact_set_hash ==
        request.model_artifact.artifact_set_sha256);
  CHECK(snapshot.active_program.invalidation_key.generations.context ==
        Domain().logical_context_generation);
  CHECK(snapshot.active_program.invalidation_key.generations.configuration ==
        request.generations.configuration);
  CHECK(snapshot.active_program.invalidation_key.generations.geometry ==
        request.generations.geometry);
  CHECK(snapshot.active_program.invalidation_key.generations.model ==
        request.generations.model);
  CHECK(snapshot.active_program.invalidation_key.generations.provider ==
        request.generations.provider);
  CHECK(snapshot.active_program.invalidation_key.generations.device ==
        request.generations.device);
  CHECK(snapshot.active_program.invalidation_key.canonical_plan_hash != 0);
  CHECK(snapshot.active_program.invalidation_key.model_artifact_hash != 0);
  CHECK(snapshot.active_program.invalidation_key.execution_domain_hash != 0);
  CHECK(snapshot.production.prepared_capability_committed);

  FakeCudaAdapter substituted;
  substituted.substitute_provider = true;
  video::PreparedOpenCudaHardwareExecutionOwner strict_owner(
      Registration(&substituted));
  const auto rejected = strict_owner.Configure(Request());
  CHECK(rejected.route ==
        video::PreparedOpenCudaHardwareRoute::legacy_compatibility);
  CHECK(!strict_owner.Snapshot().prepared);
  return true;
}

bool ExecuteCommitFailuresAndAllocationFreeFramePath() {
  FakeCudaAdapter adapter;
  video::PreparedOpenCudaHardwareExecutionOwner owner(Registration(&adapter));
  CHECK(owner.Configure(Request()).route ==
        video::PreparedOpenCudaHardwareRoute::hal_live);
  Frames frames;
  frames.input.fill(0x43);
  hal::ReasonId reason = hal::ReasonId::none;

  auto frame = owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                      16 * 3, frames.output.data(),
                                      frames.output.size(), 16 * 3, 1);
  CHECK(frame.output_ready && frame.completion_pending);
  CHECK(frame.transfers.uploads == 1 && frame.transfers.final_readbacks == 1 &&
        frame.transfers.intermediate_uploads == 0 &&
        frame.transfers.intermediate_readbacks == 0 &&
        frame.transfers.cpu_tail_boundaries == 1);
  CHECK(frames.output == frames.input);
  CHECK(owner.CommitOutput(frame.completion, true, &reason) ==
        hal::StatusId::ready);
  CHECK(owner.Snapshot().latest_committed_frame.valid);
  CHECK(owner.Snapshot().telemetry.active_backend_frames == 1);

  frames.input.fill(0x54);
  frame = owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                 16 * 3, frames.output.data(),
                                 frames.output.size(), 16 * 3, 2);
  CHECK(owner.CommitOutput(frame.completion, false, nullptr) ==
        hal::StatusId::failed);
  CHECK(owner.Snapshot().latest_committed_frame.capture_sequence == 1);

  adapter.mode = FakeCudaAdapter::ExecuteMode::optional_fail_open;
  frames.input.fill(0x65);
  frames.output.fill(0);
  frame = owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                 16 * 3, frames.output.data(),
                                 frames.output.size(), 16 * 3, 3);
  CHECK(frame.status == hal::StatusId::degraded && frame.output_ready);
  CHECK(frame.stage.status == hal::StageExecutionStatus::optional_failed_open);
  CHECK(frames.output == frames.input);
  CHECK(owner.CommitOutput(frame.completion, true, &reason) ==
        hal::StatusId::ready);
  CHECK(owner.Snapshot().latest_committed_frame.stages[0].status ==
        hal::StageExecutionStatus::optional_failed_open);
  CHECK(owner.Snapshot().telemetry.active_backend_frames == 1);

  adapter.mode = FakeCudaAdapter::ExecuteMode::success;
  const auto prepare_calls = adapter.prepare_calls;
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_release);
  bool succeeded = true;
  for (std::uint64_t sequence = 4; sequence != 68; ++sequence) {
    const auto next = owner.ExecuteHostFrame(
        frames.input.data(), frames.input.size(), 16 * 3, frames.output.data(),
        frames.output.size(), 16 * 3, sequence);
    if (!next.output_ready ||
        owner.CommitOutput(next.completion, true, &reason) !=
            hal::StatusId::ready) {
      succeeded = false;
      break;
    }
  }
  g_count_allocations.store(false, std::memory_order_release);
  CHECK(succeeded);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  CHECK(adapter.prepare_calls == prepare_calls);

  CHECK(owner.Snapshot().latest_committed_frame.valid);
  owner.Invalidate(hal::ReasonId::reconfiguration_required);
  const auto retired = owner.Snapshot();
  CHECK(!retired.prepared);
  CHECK(!retired.readiness.available);
  CHECK(retired.readiness.status == hal::StatusId::invalid);
  CHECK(!retired.latest_committed_frame.valid);
  CHECK(retired.active_program.program_id == 0);
  CHECK(retired.execution_domain.backend == hal::BackendId::invalid);
  CHECK(retired.telemetry.active_backend == hal::BackendId::invalid);
  CHECK(retired.telemetry.active_provider ==
        hal::SemanticEffectProviderId::invalid);
  CHECK(retired.telemetry.runtime_status == hal::StatusId::stale);
  CHECK(retired.telemetry.runtime_reason ==
        hal::ReasonId::reconfiguration_required);

  video::PreparedOpenCudaHardwareExecutionOwner unavailable;
  unavailable.Shutdown();
  const auto unavailable_stopped = unavailable.Snapshot();
  CHECK(unavailable_stopped.stopped);
  CHECK(unavailable_stopped.production.status == hal::StatusId::stopped);
  CHECK(unavailable_stopped.production.shutdown_requested);
  CHECK(unavailable_stopped.production.quiescent);
  CHECK(unavailable_stopped.production.driver_context_release_permitted);
  return true;
}

bool RollbackPendingReconfigurationAndResourceValidation() {
  FakeCudaAdapter adapter;
  video::PreparedOpenCudaHardwareExecutionOwner owner(Registration(&adapter));
  auto request = Request();
  CHECK(owner.Configure(request).route ==
        video::PreparedOpenCudaHardwareRoute::hal_live);
  const auto original = owner.Snapshot().active_program;

  adapter.fail_prepare = true;
  auto retained = owner.Configure(request);
  CHECK(retained.retained_previous_live_program);
  CHECK(retained.route == video::PreparedOpenCudaHardwareRoute::hal_live);
  CHECK(owner.Snapshot().active_program.program_id == original.program_id);

  auto changed = request;
  ++changed.generations.configuration;
  const auto rejected = owner.Configure(changed);
  CHECK(!rejected.retained_previous_live_program);
  CHECK(!owner.Snapshot().prepared);
  adapter.fail_prepare = false;
  CHECK(owner.Configure(request).route ==
        video::PreparedOpenCudaHardwareRoute::hal_live);

  Frames frames;
  frames.input.fill(0x76);
  auto frame = owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                      16 * 3, frames.output.data(),
                                      frames.output.size(), 16 * 3, 1);
  CHECK(frame.output_ready);
  changed = request;
  ++changed.generations.geometry;
  const auto pending = owner.Configure(changed);
  CHECK(pending.status == hal::StatusId::blocked);
  CHECK(pending.reason == hal::ReasonId::output_not_committed);
  CHECK(pending.retained_previous_live_program);
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(owner.CommitOutput(frame.completion, false, &reason) ==
        hal::StatusId::failed);

  adapter.mode = FakeCudaAdapter::ExecuteMode::foreign_output;
  frame = owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                 16 * 3, frames.output.data(),
                                 frames.output.size(), 16 * 3, 2);
  CHECK(!frame.output_ready);
  CHECK(frame.reason == hal::ReasonId::wrong_execution_domain ||
        frame.reason == hal::ReasonId::invalid_resource);

  adapter.mode = FakeCudaAdapter::ExecuteMode::wrong_generation;
  frame = owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                 16 * 3, frames.output.data(),
                                 frames.output.size(), 16 * 3, 3);
  CHECK(!frame.output_ready);
  CHECK(frame.reason == hal::ReasonId::wrong_generation ||
        frame.reason == hal::ReasonId::invalid_resource);

  FakeCudaAdapter exhausted_adapter;
  exhausted_adapter.fail_prepare = true;
  video::PreparedOpenCudaHardwareExecutionOwner exhausted(
      Registration(&exhausted_adapter));
  CHECK(exhausted.Configure(Request()).status ==
        hal::StatusId::resource_exhausted);
  CHECK(!exhausted.Snapshot().prepared);

  FakeCudaAdapter foreign_adapter;
  auto foreign_registration = Registration(&foreign_adapter);
  ++foreign_registration.execution_domain.adapter_instance_id;
  video::PreparedOpenCudaHardwareExecutionOwner foreign(foreign_registration);
  CHECK(foreign.Configure(Request()).reason ==
        hal::ReasonId::semantic_provider_unavailable);
  return true;
}

bool RequiredFailureDeviceLossAndShutdownFailClosed() {
  FakeCudaAdapter required_adapter;
  video::PreparedOpenCudaHardwareExecutionOwner required_owner(
      Registration(&required_adapter));
  CHECK(required_owner
            .Configure(
                Request(hal::SemanticEffectId::virtual_background_remove, true))
            .route == video::PreparedOpenCudaHardwareRoute::hal_live);
  required_adapter.mode = FakeCudaAdapter::ExecuteMode::required_failure;
  Frames frames;
  auto frame = required_owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 16 * 3, frames.output.data(),
      frames.output.size(), 16 * 3, 1);
  CHECK(!frame.output_ready);
  CHECK(frame.reason == hal::ReasonId::required_stage_failed);

  FakeCudaAdapter lost_adapter;
  video::PreparedOpenCudaHardwareExecutionOwner lost_owner(
      Registration(&lost_adapter));
  CHECK(lost_owner.Configure(Request()).route ==
        video::PreparedOpenCudaHardwareRoute::hal_live);
  lost_adapter.mode = FakeCudaAdapter::ExecuteMode::device_loss;
  frame = lost_owner.ExecuteHostFrame(frames.input.data(), frames.input.size(),
                                      16 * 3, frames.output.data(),
                                      frames.output.size(), 16 * 3, 1);
  CHECK(!frame.output_ready && frame.reason == hal::ReasonId::device_lost);
  CHECK(lost_owner.Snapshot().quarantined);

  FakeCudaAdapter observed_adapter;
  video::PreparedOpenCudaHardwareExecutionOwner observed_owner(
      Registration(&observed_adapter));
  CHECK(observed_owner.Configure(Request()).route ==
        video::PreparedOpenCudaHardwareRoute::hal_live);
  observed_owner.ObserveDeviceLoss();
  CHECK(observed_owner.Snapshot().quarantined);
  CHECK(observed_owner.Snapshot().production.quarantined);
  CHECK(observed_owner.Snapshot().production.device_lost);
  CHECK(!observed_owner.Snapshot().prepared);

  lost_owner.ObserveDeviceLoss();
  CHECK(lost_owner.Snapshot().quarantined);
  const auto quarantined = lost_owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 16 * 3, frames.output.data(),
      frames.output.size(), 16 * 3, 2);
  CHECK(quarantined.status == hal::StatusId::quarantined);

  lost_owner.Shutdown();
  auto snapshot = lost_owner.Snapshot();
  CHECK(snapshot.stopped);
  CHECK(snapshot.production.shutdown_requested);
  CHECK(snapshot.production.quiescent);
  CHECK(snapshot.production.driver_context_release_permitted);
  lost_owner.Shutdown();
  CHECK(lost_adapter.shutdown_calls == 1);
  CHECK(lost_owner.Configure(Request()).status == hal::StatusId::stopped);
  return true;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  if (!EligibilityMatrixFailsClosed() ||
      !ExactPreparedProgramAndStrictSubstitution() ||
      !ExecuteCommitFailuresAndAllocationFreeFramePath() ||
      !RollbackPendingReconfigurationAndResourceValidation() ||
      !RequiredFailureDeviceLossAndShutdownFailClosed()) {
    return EXIT_FAILURE;
  }
  std::cout << "Prepared Open CUDA hardware execution owner tests passed\n";
  return EXIT_SUCCESS;
}
