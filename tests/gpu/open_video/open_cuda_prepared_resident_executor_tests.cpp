#include "core/open_video/open_cuda_prepared_resident_executor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using namespace studiocast;
using RuntimeStatus = open_cuda::PreparedOpenCudaRuntimeStatus;

bool Expect(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "[FAIL] " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain(std::uint64_t adapter = 81,
                              std::uint64_t context = 7,
                              std::uint64_t stream = 0xCAFE) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10DE;
  domain.physical_device.device_id = 0x1CBB;
  domain.physical_device.stable_id[0] = 0x42;
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
  artifact.registry_generation = 3;
  artifact.model_id_sha256[0] = key;
  artifact.manifest_sha256[0] = static_cast<std::uint8_t>(key + 1);
  artifact.artifact_sha256[0] = static_cast<std::uint8_t>(key + 2);
  artifact.artifact_set_sha256[0] = static_cast<std::uint8_t>(key + 3);
  return artifact;
}

open_video::ModelPack
Pack(const open_video::ValidatedModelArtifactIdentity &artifact) {
  open_video::ModelPack pack;
  pack.id = "hermetic-matting";
  pack.display_name = "Hermetic matting";
  pack.task = "matting";
  pack.registry_generation = artifact.registry_generation;
  pack.registry_manifest_sha256 = artifact.manifest_sha256;
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

hal::ImageDescriptor HostImage(hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = 16;
  image.height = 8;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::coherent;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = 48;
  image.planes[0].size_bytes = 384;
  image.planes[0].width = 16;
  image.planes[0].height = 8;
  image.planes[0].alignment_bytes = 1;
  return image;
}

hal::ImageDescriptor DeviceImage(hal::Mutability mutability) {
  hal::ImageDescriptor image = HostImage(mutability);
  image.host_coherence = hal::HostCoherence::not_host_visible;
  image.base_alignment_bytes = 16;
  image.planes[0].alignment_bytes = 16;
  return image;
}

hal::PreparedProgram Program(hal::SemanticEffectId semantic =
                                 hal::SemanticEffectId::virtual_background_blur,
                             bool required = true, std::uint64_t id = 101) {
  const auto artifact = Artifact();
  hal::PreparedProgram program;
  program.program_id = id;
  program.owner_adapter_instance_id = 81;
  program.execution_domain = Domain();
  program.host_input = HostImage(hal::Mutability::read_only);
  program.host_output = HostImage(hal::Mutability::write_only);
  program.invalidation_key.generations = {
      .configuration = 11,
      .geometry = 12,
      .model = 13,
      .provider = 14,
      .device = 15,
      .context = 7,
  };
  program.invalidation_key.canonical_plan_hash = 0xA1;
  program.invalidation_key.model_artifact_hash = 0xA2;
  program.invalidation_key.execution_domain_hash = 0xA3;
  program.budget.persistent_bytes = 128 * 1024;
  program.budget.scratch_bytes = 64 * 1024;
  program.budget.upload_staging_bytes = 4096;
  program.budget.readback_staging_bytes = 4096;
  program.budget.resource_count = 32;
  program.budget.in_flight_slots = 1;
  auto &stage = program.stages[0];
  stage.stage_id = 1;
  stage.semantic = semantic;
  stage.provider = hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  stage.backend = hal::BackendId::cuda;
  stage.execution_domain = program.execution_domain;
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image = DeviceImage(hal::Mutability::read_only);
  stage.output.kind = hal::StageIoKind::image;
  stage.output.image = DeviceImage(hal::Mutability::read_write);
  stage.model.model_id_hash = artifact.model_id_sha256;
  stage.model.artifact_set_hash = artifact.artifact_set_sha256;
  stage.model.artifact_generation = artifact.registry_generation;
  stage.required = required;
  stage.failure_action = required
                             ? hal::FailureAction::fail_closed
                             : hal::FailureAction::fail_open_preserve_input;
  stage.circuit_breaker.consecutive_failure_limit = 2;
  stage.circuit_breaker.cooldown_frames = 1;
  stage.circuit_breaker.action =
      required ? hal::CircuitBreakerAction::quarantine_program
               : hal::CircuitBreakerAction::disable_stage;
  stage.scratch_bytes = 4096;
  program.stage_count = 1;
  program.cpu_tails[0].id = hal::CpuTailId::writer_copy;
  program.cpu_tails[0].after_stage_id = stage.stage_id;
  program.cpu_tails[0].required = true;
  program.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  program.cpu_tails[0].boundary_index = 1;
  program.cpu_tail_count = 1;
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = true;
  program.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  program.strict_provider_selection = true;
  program.requested_execution_domain = program.execution_domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;
  return program;
}

open_cuda::OpenCudaPreparedResidentExecutor::Options
Options(open_cuda::IPreparedOpenCudaRuntime *runtime) {
  const auto artifact = Artifact();
  open_cuda::OpenCudaPreparedResidentExecutor::Options options;
  options.prepared_stream =
      reinterpret_cast<cuda::CUstream>(std::uintptr_t{0xCAFE});
  options.model_pack = Pack(artifact);
  options.model_artifact = artifact;
  options.cuda_device_id = 0;
  options.adapter_instance_id = 81;
  options.stream_identity = 0xCAFE;
  options.configuration_generation = 11;
  options.model_preprocess_generation = 13;
  options.model_artifact_hash = 0xA2;
  options.executor_owns_stream = true;
  options.strength = 8;
  options.runtime = runtime;
  return options;
}

hal::HostFrameBinding HostBinding(const hal::PreparedProgram &program,
                                  const std::byte *input, std::byte *output,
                                  std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = program.host_input;
  binding.output_descriptor = program.host_output;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  binding.input_planes[0].read_data = input;
  binding.input_planes[0].span_bytes = 384;
  binding.input_planes[0].row_stride_bytes = 48;
  binding.output_planes[0].write_data = output;
  binding.output_planes[0].span_bytes = 384;
  binding.output_planes[0].row_stride_bytes = 48;
  return binding;
}

class FakeRuntime final : public open_cuda::IPreparedOpenCudaRuntime {
public:
  struct Slot {
    bool active = false;
    open_cuda::PreparedOpenCudaRuntimeHandle handle{};
    hal::PreparedProgram program{};
  };

  RuntimeStatus execute_status = RuntimeStatus::executed;
  bool fail_prepare = false;
  bool oversized_budget = false;
  bool incomplete_evidence = false;
  bool malformed_plan = false;
  int forbidden_frame_work = 0;
  bool no_analysis = false;
  bool pending_destroy = false;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;
  open_cuda::PreparedOpenCudaRuntimeTelemetry telemetry{};
  open_cuda::PreparedOpenCudaShutdownSnapshot shutdown_snapshot{};
  open_cuda::PreparedOpenCudaVbPlan prepared_plan{};
  open_cuda::PreparedOpenCudaOperationId last_failed_operation =
      open_cuda::PreparedOpenCudaOperationId::invalid;

  open_cuda::PreparedOpenCudaRuntimePrepareResult
  Prepare(const hal::PreparedProgram &program,
          const open_cuda::PreparedOpenCudaProductionConfig &config) noexcept
      override {
    ++prepare_calls;
    open_cuda::PreparedOpenCudaRuntimePrepareResult result;
    if (fail_prepare) {
      result.status = RuntimeStatus::resource_exhausted;
      result.reason = hal::ReasonId::resource_exhausted;
      return result;
    }
    Slot *slot = nullptr;
    for (auto &candidate : slots_) {
      if (!candidate.active) {
        slot = &candidate;
        break;
      }
    }
    if (!slot) {
      result.status = RuntimeStatus::resource_exhausted;
      result.reason = hal::ReasonId::resource_exhausted;
      return result;
    }
    result.status = RuntimeStatus::ready;
    result.reason = hal::ReasonId::none;
    result.handle.handle_id = next_handle_++;
    result.handle.program_id = program.program_id;
    result.handle.invalidation_key = program.invalidation_key;
    result.handle.execution_domain = program.execution_domain;
    result.handle.opaque[0] = result.handle.handle_id + 100;
    result.evidence.status = RuntimeStatus::ready;
    result.evidence.semantic = program.stages[0].semantic;
    result.evidence.provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    result.evidence.execution_domain = program.execution_domain;
    result.evidence.model_artifact = config.model_artifact;
    open_cuda::PreparedOpenCudaVbPlanInput plan_input;
    plan_input.semantic = program.stages[0].semantic;
    plan_input.frame_width = program.host_input.width;
    plan_input.frame_height = program.host_input.height;
    plan_input.model_width =
        static_cast<std::uint64_t>(config.model_pack->matting->input.width);
    plan_input.model_height =
        static_cast<std::uint64_t>(config.model_pack->matting->input.height);
    plan_input.requested_strength = config.strength;
    plan_input.input_pitch_bytes =
        program.stages[0].input.image.planes[0].row_stride_bytes;
    plan_input.output_pitch_bytes =
        program.stages[0].output.image.planes[0].row_stride_bytes;
    plan_input.model_alpha_capacity_bytes =
        plan_input.model_width * plan_input.model_height * sizeof(float);
    plan_input.input_tensor_capacity_bytes =
        plan_input.model_alpha_capacity_bytes * 3;
    plan_input.resized_alpha_pitch_bytes =
        plan_input.frame_width * sizeof(float);
    plan_input.alpha_scratch_pitch_bytes =
        plan_input.frame_width * sizeof(float);
    plan_input.alpha_feather_pitch_bytes =
        plan_input.frame_width * sizeof(float);
    plan_input.blur_scratch_pitch_bytes = plan_input.input_pitch_bytes;
    plan_input.blurred_pitch_bytes = plan_input.input_pitch_bytes;
    plan_input.upload_staging_bytes = program.host_input.planes[0].size_bytes;
    plan_input.readback_staging_bytes =
        program.host_output.planes[0].size_bytes;
    plan_input.available_budget = program.budget;
    result.evidence.geometry_plan =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(plan_input);
    if (malformed_plan)
      result.evidence.geometry_plan.launches[0].operation =
          open_cuda::PreparedOpenCudaOperationId::composite_solid;
    prepared_plan = result.evidence.geometry_plan;
    result.evidence.reserved_budget =
        result.evidence.geometry_plan.required_budget;
    if (oversized_budget)
      result.evidence.reserved_budget.persistent_bytes =
          program.budget.persistent_bytes + 1;
    result.evidence.strict_cuda_only = true;
    result.evidence.user_stream_bound = true;
    result.evidence.cpu_provider_fallback_disabled = true;
    result.evidence.reusable_resources_prepared = true;
    result.evidence.setup_warmup_complete = true;
    result.evidence.inference_only_frame_path = true;
    result.evidence.device_preprocess = true;
    result.evidence.device_postprocess = !incomplete_evidence;
    result.evidence.one_upload_one_readback = true;
    result.evidence.no_intermediate_crossings = true;
    result.evidence.single_in_flight = true;
    shutdown_snapshot.stream_owned = config.executor_owns_stream;
    slot->active = true;
    slot->handle = result.handle;
    slot->program = program;
    return result;
  }

  open_cuda::PreparedOpenCudaRuntimeFrameResult ExecuteFrame(
      const open_cuda::PreparedOpenCudaRuntimeHandle &handle,
      const open_cuda::PreparedOpenCudaRuntimeFrame &frame) noexcept override {
    ++execute_calls;
    open_cuda::PreparedOpenCudaRuntimeFrameResult result;
    const Slot *slot = Find(handle);
    if (!slot) {
      result.status = RuntimeStatus::foreign_handle;
      result.reason = hal::ReasonId::stale_program;
      return result;
    }
    if (forbidden_frame_work == 1)
      ++telemetry.allocation_requests;
    else if (forbidden_frame_work == 2)
      ++telemetry.provider_probes;
    else if (forbidden_frame_work == 3)
      ++telemetry.model_scans;
    else if (forbidden_frame_work == 4)
      ++telemetry.manifest_parses;
    else if (forbidden_frame_work == 5)
      ++telemetry.session_constructions;
    else if (forbidden_frame_work == 6)
      ++telemetry.json_serializations;
    result.status = execute_status;
    if (frame.stage_disabled)
      result.status = RuntimeStatus::executed;
    result.reason = result.status == RuntimeStatus::optional_failed_open
                        ? hal::ReasonId::optional_stage_failed_open
                        : (result.status == RuntimeStatus::required_failed
                               ? hal::ReasonId::required_stage_failed
                               : (result.status == RuntimeStatus::device_lost
                                      ? hal::ReasonId::device_lost
                                      : hal::ReasonId::none));
    if (result.status == RuntimeStatus::optional_failed_open) {
      result.stage_failure_status = RuntimeStatus::inference_failed;
      result.stage_failure_reason =
          hal::ReasonId::semantic_provider_unavailable;
      result.failed_operation =
          open_cuda::PreparedOpenCudaOperationId::inference;
    } else if (result.status == RuntimeStatus::required_failed) {
      result.stage_failure_status = RuntimeStatus::postprocess_failed;
      result.stage_failure_reason = hal::ReasonId::transfer_contract_violated;
      result.failed_operation =
          slot->program.stages[0].semantic ==
                  hal::SemanticEffectId::virtual_background_remove
              ? open_cuda::PreparedOpenCudaOperationId::composite_solid
              : open_cuda::PreparedOpenCudaOperationId::composite_background;
    } else if (result.status == RuntimeStatus::device_lost) {
      result.stage_failure_status = RuntimeStatus::device_lost;
      result.stage_failure_reason = hal::ReasonId::device_lost;
      result.failed_operation =
          open_cuda::PreparedOpenCudaOperationId::completion_wait;
    }
    last_failed_operation = result.failed_operation;
    const bool completed = result.status == RuntimeStatus::executed ||
                           result.status == RuntimeStatus::optional_failed_open;
    if (completed) {
      std::memcpy(frame.output, frame.input,
                  std::min(frame.input_span_bytes, frame.output_span_bytes));
      if (!frame.stage_disabled && result.status == RuntimeStatus::executed &&
          frame.output_span_bytes != 0) {
        frame.output[0] ^= std::byte{0x7F};
      }
      result.output_valid = true;
      result.valid_input_preserved =
          frame.stage_disabled ||
          result.status == RuntimeStatus::optional_failed_open;
      result.transfers.uploads = 1;
      result.transfers.final_readbacks = 1;
      result.transfers.synchronizations = 1;
      result.transfers.cpu_tail_boundaries = 1;
      result.stage_elapsed_ns = frame.stage_disabled ? 0 : 1234;
      result.frame_elapsed_ns = 2345;
      result.analysis_ran = !frame.stage_disabled && !no_analysis;
      result.inference_only_path = !frame.stage_disabled;
    } else if (result.status == RuntimeStatus::required_failed) {
      result.transfers.uploads = 1;
      result.transfers.final_readbacks = 1;
      result.transfers.synchronizations = 1;
      result.stage_elapsed_ns = 1234;
      result.analysis_ran = true;
      result.inference_only_path = true;
    } else if (result.status == RuntimeStatus::device_lost) {
      result.analysis_ran = true;
      result.inference_only_path = true;
    }
    return result;
  }

  RuntimeStatus
  ResetTemporalState(const open_cuda::PreparedOpenCudaRuntimeHandle &handle,
                     hal::ReasonId) noexcept override {
    ++reset_calls;
    return Find(handle) ? RuntimeStatus::ready : RuntimeStatus::foreign_handle;
  }

  RuntimeStatus DestroyPrepared(
      open_cuda::PreparedOpenCudaRuntimeHandle *handle) noexcept override {
    ++destroy_calls;
    if (!handle)
      return RuntimeStatus::foreign_handle;
    for (auto &slot : slots_) {
      if (slot.active && slot.handle.handle_id == handle->handle_id) {
        *handle = {};
        if (pending_destroy)
          return RuntimeStatus::completion_pending;
        slot = {};
        return RuntimeStatus::ready;
      }
    }
    *handle = {};
    return RuntimeStatus::foreign_handle;
  }

  open_cuda::PreparedOpenCudaShutdownSnapshot
  RequestShutdown() noexcept override {
    ++shutdown_calls;
    ++shutdown_snapshot.shutdown_attempts;
    shutdown_snapshot.shutdown_requested = true;
    std::uint32_t pending_slots = 0;
    for (auto &slot : slots_) {
      if (!slot.active)
        continue;
      if (pending_destroy) {
        ++pending_slots;
        continue;
      }
      slot = {};
    }
    if (pending_slots == 0 && shutdown_snapshot.stream_owned &&
        !shutdown_snapshot.stream_destroyed) {
      shutdown_snapshot.stream_destroyed = true;
      ++shutdown_snapshot.stream_destructions;
    }
    shutdown_snapshot.pending_prepared_slots = pending_slots;
    shutdown_snapshot.quiescent =
        pending_slots == 0 &&
        (!shutdown_snapshot.stream_owned || shutdown_snapshot.stream_destroyed);
    shutdown_snapshot.driver_context_release_permitted =
        shutdown_snapshot.quiescent;
    shutdown_snapshot.status = shutdown_snapshot.quiescent
                                   ? RuntimeStatus::ready
                                   : RuntimeStatus::completion_pending;
    return shutdown_snapshot;
  }

  open_cuda::PreparedOpenCudaShutdownSnapshot
  ShutdownSnapshot() const noexcept override {
    return shutdown_snapshot;
  }

  open_cuda::PreparedOpenCudaRuntimeTelemetry
  Telemetry() const noexcept override {
    return telemetry;
  }

private:
  const Slot *
  Find(const open_cuda::PreparedOpenCudaRuntimeHandle &handle) const {
    for (const auto &slot : slots_) {
      if (slot.active && slot.handle.handle_id == handle.handle_id &&
          slot.handle.program_id == handle.program_id &&
          slot.handle.invalidation_key == handle.invalidation_key &&
          slot.handle.execution_domain == handle.execution_domain &&
          slot.handle.opaque == handle.opaque) {
        return &slot;
      }
    }
    return nullptr;
  }

  std::array<Slot, 8> slots_{};
  std::uint64_t next_handle_ = 1;
};

struct Fixture {
  explicit Fixture(hal::SemanticEffectId semantic =
                       hal::SemanticEffectId::virtual_background_blur,
                   bool required = true)
      : program(Program(semantic, required)), executor(Options(&runtime)),
        adapter(81, executor) {}

  FakeRuntime runtime;
  hal::PreparedProgram program;
  open_cuda::OpenCudaPreparedResidentExecutor executor;
  open_cuda::ResidentAdapter adapter;
  hal::PreparedProgramSession session;
  std::array<std::byte, 384> input{};
  std::array<std::byte, 384> output{};
};

bool Configure(Fixture *fixture) {
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto status = fixture->session.Configure(
      fixture->program, fixture->adapter, &readiness, &reason);
  return Expect(
      status == hal::StatusId::degraded &&
          readiness.reason == hal::ReasonId::cpu_tail_required &&
          readiness.provider ==
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
      "strict prepared program should configure with its "
      "declared writer-copy CPU tail");
}

hal::ExecuteResult ExecuteBound(Fixture *fixture, std::uint64_t sequence) {
  for (std::size_t index = 0; index < fixture->input.size(); ++index)
    fixture->input[index] = static_cast<std::byte>(index & 0xFFu);
  fixture->output.fill(std::byte{0xEE});
  hal::HostFrameTokens tokens;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto bind = fixture->session.BindHostFrame(
      HostBinding(fixture->program, fixture->input.data(),
                  fixture->output.data(), sequence),
      &tokens, &reason);
  if (!Expect(bind == hal::StatusId::ready,
              "host frame should bind without runtime work")) {
    return {};
  }
  hal::ExecuteRequest request;
  request.invalidation_key = fixture->program.invalidation_key;
  request.execution_domain = fixture->program.execution_domain;
  request.capture_sequence = sequence;
  request.input = tokens.input;
  return fixture->session.Execute(request);
}

bool ResolveAndCommit(Fixture *fixture, const hal::ExecuteResult &result,
                      bool write_succeeded = true) {
  hal::HostFrameOutputResolution resolution;
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Expect(fixture->session.ResolveHostFrameOutput(
                       result.completion, result.output, &resolution,
                       &reason) == hal::StatusId::ready,
                   "completed host output should resolve");
  hal::OutputCommit commit;
  commit.completion = result.completion;
  commit.output_write_succeeded = write_succeeded;
  const auto status = fixture->session.CommitOutput(commit, &reason);
  ok = Expect(status == (write_succeeded ? hal::StatusId::ready
                                         : hal::StatusId::failed),
              "writer commit should remain authoritative") &&
       ok;
  return ok;
}

open_cuda::PreparedOpenCudaVbPlanInput
PlanningInput(hal::SemanticEffectId semantic, int strength) {
  open_cuda::PreparedOpenCudaVbPlanInput input;
  input.semantic = semantic;
  input.frame_width = 17;
  input.frame_height = 9;
  input.model_width = 8;
  input.model_height = 4;
  input.requested_strength = strength;
  input.input_pitch_bytes = 64;
  input.output_pitch_bytes = 80;
  input.model_alpha_capacity_bytes = 128;
  input.input_tensor_capacity_bytes = 384;
  input.resized_alpha_pitch_bytes = 96;
  input.alpha_scratch_pitch_bytes = 112;
  input.alpha_feather_pitch_bytes = 128;
  input.blur_scratch_pitch_bytes = 64;
  input.blurred_pitch_bytes = 80;
  input.upload_staging_bytes = 459;
  input.readback_staging_bytes = 459;
  input.available_budget.persistent_bytes = 1'000'000;
  input.available_budget.scratch_bytes = 1'000'000;
  input.available_budget.upload_staging_bytes = 1'000'000;
  input.available_budget.readback_staging_bytes = 1'000'000;
  input.available_budget.resource_count = 100;
  input.available_budget.in_flight_slots = 1;
  return input;
}

bool TestPreparedGeometryStrengthLaunchesAndBudgets() {
  using Operation = open_cuda::PreparedOpenCudaOperationId;
  bool ok = true;
  struct Threshold {
    int requested = 0;
    std::uint32_t strength = 0;
    std::uint32_t feather = 0;
  };
  for (const auto threshold :
       {Threshold{-10, 1, 0}, Threshold{1, 1, 0}, Threshold{15, 15, 0},
        Threshold{16, 16, 1}, Threshold{63, 63, 3}, Threshold{64, 64, 4},
        Threshold{100, 64, 4}}) {
    const auto blur = open_cuda::PlanPreparedOpenCudaVirtualBackground(
        PlanningInput(hal::SemanticEffectId::virtual_background_blur,
                      threshold.requested));
    const std::uint32_t expected_launches = threshold.feather == 0 ? 4 : 6;
    ok = Expect(
             blur.ready() && blur.frame_width == 17 && blur.frame_height == 9 &&
                 blur.model_width == 8 && blur.model_height == 4 &&
                 blur.strength == threshold.strength &&
                 blur.alpha_feather_radius == threshold.feather &&
                 blur.launch_count == expected_launches &&
                 blur.launches[0].operation == Operation::alpha_resize &&
                 blur.launches[0].grid_x == 2 && blur.launches[0].grid_y == 1 &&
                 blur.launches[0].block_x == 16 &&
                 blur.launches[0].block_y == 16 &&
                 blur.launches[expected_launches - 3].operation ==
                     Operation::background_blur_horizontal &&
                 blur.launches[expected_launches - 2].operation ==
                     Operation::background_blur_vertical &&
                 blur.launches[expected_launches - 1].operation ==
                     Operation::composite_background,
             "blur strength thresholds must prepare exact resize, optional "
             "feather, blur, and composite launch geometry") &&
         ok;
    if (threshold.feather != 0) {
      ok = Expect(blur.launches[1].operation ==
                          Operation::alpha_feather_horizontal &&
                      blur.launches[2].operation ==
                          Operation::alpha_feather_vertical,
                  "enabled alpha feather must be a prepared separable pair") &&
           ok;
    }

    const auto remove = open_cuda::PlanPreparedOpenCudaVirtualBackground(
        PlanningInput(hal::SemanticEffectId::virtual_background_remove,
                      threshold.requested));
    const std::uint32_t expected_remove_launches =
        threshold.feather == 0 ? 2 : 4;
    ok = Expect(remove.ready() && remove.strength == threshold.strength &&
                    remove.alpha_feather_radius == threshold.feather &&
                    remove.launch_count == expected_remove_launches &&
                    remove.launches[0].operation == Operation::alpha_resize &&
                    remove.launches[expected_remove_launches - 1].operation ==
                        Operation::composite_solid,
                "remove strength must affect feather only and retain exact "
                "solid composite ordering") &&
         ok;
  }

  const auto blur = open_cuda::PlanPreparedOpenCudaVirtualBackground(
      PlanningInput(hal::SemanticEffectId::virtual_background_blur, 16));
  ok = Expect(blur.required_budget.persistent_bytes == 2672 &&
                  blur.required_budget.scratch_bytes == 3456 &&
                  blur.required_budget.upload_staging_bytes == 459 &&
                  blur.required_budget.readback_staging_bytes == 459 &&
                  blur.required_budget.resource_count == 21 &&
                  blur.required_budget.in_flight_slots == 1,
              "blur budget must use exact prepared pitches and capacities") &&
       ok;
  const auto remove = open_cuda::PlanPreparedOpenCudaVirtualBackground(
      PlanningInput(hal::SemanticEffectId::virtual_background_remove, 16));
  ok = Expect(remove.required_budget.persistent_bytes == 2672 &&
                  remove.required_budget.scratch_bytes == 2160 &&
                  remove.required_budget.resource_count == 19,
              "remove budget must include every prepared alpha buffer but no "
              "RGB blur buffers") &&
       ok;
  {
    auto input =
        PlanningInput(hal::SemanticEffectId::virtual_background_remove, 8);
    input.model_width = input.frame_width;
    input.model_height = input.frame_height;
    input.model_alpha_capacity_bytes =
        input.model_width * input.model_height * sizeof(float);
    input.input_tensor_capacity_bytes = input.model_alpha_capacity_bytes * 3;
    const auto same_geometry =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(input);
    ok = Expect(same_geometry.ready() && same_geometry.launches[0].operation ==
                                             Operation::alpha_resize,
                "alpha resize must remain explicit even when model and frame "
                "geometry happen to match") &&
         ok;
  }

  {
    auto input =
        PlanningInput(hal::SemanticEffectId::virtual_background_blur, 8);
    input.frame_width = std::numeric_limits<std::uint64_t>::max();
    const auto overflow =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(input);
    ok = Expect(overflow.status == RuntimeStatus::resource_exhausted &&
                    overflow.reason == hal::ReasonId::resource_exhausted,
                "geometry arithmetic overflow must fail closed") &&
         ok;
  }
  {
    auto input =
        PlanningInput(hal::SemanticEffectId::virtual_background_blur, 8);
    input.frame_width = 1;
    input.frame_height = 1;
    input.model_width = 1;
    input.model_height = 1;
    input.input_pitch_bytes = std::numeric_limits<std::uint64_t>::max();
    input.output_pitch_bytes = std::numeric_limits<std::uint64_t>::max();
    input.model_alpha_capacity_bytes = 4;
    input.input_tensor_capacity_bytes = 12;
    input.resized_alpha_pitch_bytes = 4;
    input.alpha_scratch_pitch_bytes = 4;
    input.alpha_feather_pitch_bytes = 4;
    input.blur_scratch_pitch_bytes = 3;
    input.blurred_pitch_bytes = 3;
    input.upload_staging_bytes = 3;
    input.readback_staging_bytes = 3;
    input.available_budget.persistent_bytes =
        std::numeric_limits<std::uint64_t>::max();
    input.available_budget.scratch_bytes =
        std::numeric_limits<std::uint64_t>::max();
    input.available_budget.upload_staging_bytes =
        std::numeric_limits<std::uint64_t>::max();
    input.available_budget.readback_staging_bytes =
        std::numeric_limits<std::uint64_t>::max();
    const auto overflow =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(input);
    ok = Expect(overflow.status == RuntimeStatus::resource_exhausted &&
                    overflow.reason == hal::ReasonId::resource_exhausted,
                "prepared capacity addition overflow must fail closed") &&
         ok;
  }
  {
    auto input =
        PlanningInput(hal::SemanticEffectId::virtual_background_blur, 8);
    input.available_budget.scratch_bytes = 1;
    const auto exhausted =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(input);
    ok = Expect(exhausted.status == RuntimeStatus::resource_exhausted &&
                    exhausted.reason == hal::ReasonId::resource_exhausted,
                "exact prepared capacity above budget must fail setup") &&
         ok;
  }
  {
    auto input =
        PlanningInput(hal::SemanticEffectId::virtual_background_blur, 8);
    input.resized_alpha_pitch_bytes = 1;
    const auto malformed =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(input);
    ok = Expect(malformed.status == RuntimeStatus::malformed_contract &&
                    malformed.reason == hal::ReasonId::malformed_descriptor,
                "undersized prepared pitch must fail as malformed") &&
         ok;
  }
  return ok;
}

bool TestAcceptedBlurAndRemoveThroughSession() {
  bool ok = true;
  for (const auto semantic :
       {hal::SemanticEffectId::virtual_background_blur,
        hal::SemanticEffectId::virtual_background_remove}) {
    Fixture fixture(semantic);
    ok = Configure(&fixture) && ok;
    const auto result = ExecuteBound(&fixture, 1);
    ok = Expect(result.status == hal::StatusId::executed &&
                    result.output_valid && result.stage_count == 1 &&
                    result.stages[0].status ==
                        hal::StageExecutionStatus::succeeded &&
                    result.transfers.uploads == 1 &&
                    result.transfers.final_readbacks == 1 &&
                    result.transfers.intermediate_readbacks == 0 &&
                    result.transfers.intermediate_uploads == 0 &&
                    result.transfers.synchronizations == 1 &&
                    result.transfers.cpu_tail_boundaries == 1 &&
                    result.frame_elapsed_ns >= result.stages[0].elapsed_ns &&
                    fixture.output[0] == (fixture.input[0] ^ std::byte{0x7F}),
                "blur/remove should execute one exact resident island") &&
         ok;
    ok = Expect(fixture.session.telemetry().active_backend ==
                        hal::BackendId::invalid &&
                    !fixture.session.latest_committed_frame().valid,
                "execution alone must not attribute active CUDA") &&
         ok;
    ok = ResolveAndCommit(&fixture, result) && ok;
    ok = Expect(fixture.session.telemetry().active_backend ==
                        hal::BackendId::cuda &&
                    fixture.session.telemetry().active_provider ==
                        hal::SemanticEffectProviderId::
                            open_video_onnxruntime_cuda &&
                    fixture.session.telemetry().active_backend_frames == 1 &&
                    fixture.session.latest_committed_frame().valid,
                "successful writer commit should attribute exact backend") &&
         ok;
  }
  return ok;
}

bool TestOptionalFailurePreservesInputWithoutAttribution() {
  Fixture fixture(hal::SemanticEffectId::virtual_background_blur, false);
  bool ok = Configure(&fixture);
  fixture.runtime.execute_status = RuntimeStatus::optional_failed_open;
  const auto result = ExecuteBound(&fixture, 1);
  ok = Expect(result.status == hal::StatusId::degraded &&
                  result.reason == hal::ReasonId::optional_stage_failed_open &&
                  result.stages[0].status ==
                      hal::StageExecutionStatus::optional_failed_open &&
                  result.stages[0].reason ==
                      hal::ReasonId::semantic_provider_unavailable &&
                  result.stages[0].valid_output_preserved &&
                  fixture.output == fixture.input &&
                  result.transfers.uploads == 1 &&
                  result.transfers.final_readbacks == 1 &&
                  result.transfers.synchronizations == 1 &&
                  result.transfers.cpu_tail_boundaries == 1 &&
                  fixture.runtime.last_failed_operation ==
                      open_cuda::PreparedOpenCudaOperationId::inference,
              "optional failure should preserve exact valid input") &&
       ok;
  ok = ResolveAndCommit(&fixture, result) && ok;
  ok = Expect(fixture.session.telemetry().active_backend ==
                      hal::BackendId::invalid &&
                  fixture.session.telemetry().optional_fail_open_stages == 1,
              "optional fail-open commit must not claim backend success") &&
       ok;
  return ok;
}

bool TestRequiredFailureDeviceLossAndForbiddenFrameWork() {
  bool ok = true;
  {
    Fixture fixture;
    ok = Configure(&fixture) && ok;
    fixture.runtime.execute_status = RuntimeStatus::required_failed;
    const auto result = ExecuteBound(&fixture, 1);
    ok = Expect(result.status == hal::StatusId::failed &&
                    result.reason == hal::ReasonId::required_stage_failed &&
                    !result.output_valid &&
                    result.stages[0].status ==
                        hal::StageExecutionStatus::required_failed &&
                    result.stages[0].reason ==
                        hal::ReasonId::transfer_contract_violated &&
                    fixture.runtime.last_failed_operation ==
                        open_cuda::PreparedOpenCudaOperationId::
                            composite_background &&
                    fixture.output[0] == std::byte{0xEE},
                "required failure must not publish or attribute output") &&
         ok;
  }
  {
    Fixture fixture(hal::SemanticEffectId::virtual_background_remove);
    ok = Configure(&fixture) && ok;
    fixture.runtime.execute_status = RuntimeStatus::required_failed;
    const auto result = ExecuteBound(&fixture, 1);
    ok = Expect(result.status == hal::StatusId::failed &&
                    fixture.runtime.last_failed_operation ==
                        open_cuda::PreparedOpenCudaOperationId::composite_solid,
                "remove failure must retain exact solid-composite "
                "attribution") &&
         ok;
  }
  {
    Fixture fixture;
    ok = Configure(&fixture) && ok;
    fixture.runtime.execute_status = RuntimeStatus::device_lost;
    const auto result = ExecuteBound(&fixture, 1);
    if (result.status != hal::StatusId::quarantined) {
      std::cerr << "[DETAIL] device status="
                << static_cast<std::uint32_t>(result.status)
                << " reason=" << static_cast<std::uint32_t>(result.reason)
                << " stage_count=" << result.stage_count
                << " session_quarantined=" << fixture.session.quarantined()
                << '\n';
    }
    ok = Expect(result.status == hal::StatusId::quarantined &&
                    result.reason == hal::ReasonId::device_lost &&
                    fixture.session.quarantined(),
                "device loss should quarantine the prepared program") &&
         ok;
  }
  for (int forbidden_work = 1; forbidden_work <= 6; ++forbidden_work) {
    Fixture fixture;
    ok = Configure(&fixture) && ok;
    fixture.runtime.forbidden_frame_work = forbidden_work;
    const auto result = ExecuteBound(&fixture, 1);
    ok = Expect(result.status == hal::StatusId::failed && !result.output_valid,
                "frame allocation/probe/scan/manifest/session/JSON evidence "
                "must fail the adapter contract") &&
         ok;
  }
  return ok;
}

bool TestModelAndCaptureGeometryMayDiffer() {
  FakeRuntime runtime;
  auto options = Options(&runtime);
  options.model_pack.matting->input.width = 8;
  options.model_pack.matting->input.height = 4;
  options.model_pack.matting->output.width = 8;
  options.model_pack.matting->output.height = 4;
  options.strength = 100;
  open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
  open_cuda::ResidentAdapter adapter(81, executor);
  hal::PreparedProgramSession session;
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto status =
      session.Configure(Program(), adapter, &readiness, &reason);
  return Expect(
      status == hal::StatusId::degraded && runtime.prepare_calls == 1 &&
          runtime.shutdown_snapshot.stream_owned && session.prepared() &&
          runtime.prepared_plan.frame_width == 16 &&
          runtime.prepared_plan.frame_height == 8 &&
          runtime.prepared_plan.model_width == 8 &&
          runtime.prepared_plan.model_height == 4 &&
          runtime.prepared_plan.strength == 64 &&
          runtime.prepared_plan.alpha_feather_radius == 4,
      "ordinary capture geometry must preprocess to independent model "
      "geometry and clamp strength during prepared planning");
}

bool TestEligibilityFailsClosedBeforeRuntimeSetup() {
  bool ok = true;
  for (int mode = 0; mode < 11; ++mode) {
    FakeRuntime runtime;
    auto program = Program();
    auto options = Options(&runtime);
    if (mode == 0)
      program.stages[0].semantic =
          hal::SemanticEffectId::virtual_background_replace;
    else if (mode == 1)
      program.stages[0].semantic = hal::SemanticEffectId::virtual_key_light;
    else if (mode == 2) {
      program.stages[1] = program.stages[0];
      program.stages[1].stage_id = 2;
      program.stage_count = 2;
    } else if (mode == 3) {
      program.host_output.width = 8;
      program.host_output.planes[0].width = 8;
      program.host_output.planes[0].row_stride_bytes = 24;
      program.host_output.planes[0].size_bytes = 192;
    } else if (mode == 4) {
      program.strict_provider_selection = false;
    } else if (mode == 5) {
      program.requested_provider =
          hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
    } else if (mode == 6) {
      ++program.execution_domain.queue_or_stream_identity;
      program.requested_execution_domain = program.execution_domain;
      program.stages[0].execution_domain = program.execution_domain;
    } else if (mode == 7) {
      ++program.stages[0].model.artifact_generation;
    } else if (mode == 8) {
      program.temporal.temporal = true;
      program.temporal.history_frames = 1;
    } else if (mode == 9) {
      options.model_pack.matting->input.width = 0;
    } else {
      options.prepared_stream =
          reinterpret_cast<cuda::CUstream>(std::uintptr_t{0xBEEF});
    }
    open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status =
        session.Configure(program, adapter, &readiness, &reason);
    ok = Expect(status != hal::StatusId::ready &&
                    status != hal::StatusId::degraded &&
                    runtime.prepare_calls == 0,
                "unsupported/mixed/scaled/non-strict/foreign/stale "
                "program must fail before runtime setup") &&
         ok;
  }
  {
    FakeRuntime runtime;
    auto options = Options(&runtime);
    options.prepared_stream =
        reinterpret_cast<cuda::CUstream>(std::uintptr_t{0xBEEF});
    open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status =
        session.Configure(Program(), adapter, &readiness, &reason);
    ok = Expect(status == hal::StatusId::blocked &&
                    reason == hal::ReasonId::wrong_execution_domain &&
                    runtime.prepare_calls == 0,
                "execution-domain stream identity must be the exact owned "
                "CUstream, not an unrelated logical number") &&
         ok;
  }
  return ok;
}

bool TestBudgetEvidenceRollbackAndOldPlanRetention() {
  bool ok = true;
  {
    FakeRuntime runtime;
    runtime.malformed_plan = true;
    auto options = Options(&runtime);
    open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status =
        session.Configure(Program(), adapter, &readiness, &reason);
    ok = Expect(status == hal::StatusId::blocked &&
                    reason == hal::ReasonId::capability_evidence_incomplete &&
                    runtime.destroy_calls == 1,
                "malformed prepared launch order must be rejected and rolled "
                "back") &&
         ok;
  }
  {
    FakeRuntime runtime;
    runtime.oversized_budget = true;
    auto options = Options(&runtime);
    open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status =
        session.Configure(Program(), adapter, &readiness, &reason);
    ok = Expect(status == hal::StatusId::blocked &&
                    reason == hal::ReasonId::capability_evidence_incomplete &&
                    runtime.destroy_calls == 1,
                "oversized runtime evidence should be rejected and "
                "destroyed") &&
         ok;
  }
  {
    Fixture fixture;
    ok = Configure(&fixture) && ok;
    fixture.runtime.fail_prepare = true;
    auto replacement = fixture.program;
    replacement.program_id = 102;
    replacement.invalidation_key.canonical_plan_hash = 0xB1;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status = fixture.session.Configure(replacement, fixture.adapter,
                                                  &readiness, &reason);
    ok = Expect(status == hal::StatusId::resource_exhausted &&
                    fixture.session.prepared() &&
                    fixture.session.program().program_id ==
                        fixture.program.program_id &&
                    fixture.session.telemetry().setup_rollbacks == 1,
                "failed candidate should retain the old exact program") &&
         ok;
    fixture.runtime.fail_prepare = false;
    const auto result = ExecuteBound(&fixture, 1);
    ok = Expect(result.status == hal::StatusId::executed,
                "retained old program should remain executable") &&
         ok;
  }
  return ok;
}

bool TestStaleForeignBindingReconfigurationAndReset() {
  FakeRuntime runtime;
  auto options = Options(&runtime);
  open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
  const auto program = Program();
  auto prepared = executor.Prepare(program);
  bool ok = Expect(prepared.status == hal::StatusId::ready,
                   "direct executor setup should succeed");
  auto foreign = prepared.handle;
  ++foreign.handle_id;
  hal::ExecuteRequest empty_request;
  auto result = executor.Execute(foreign, empty_request);
  ok = Expect(result.status == hal::StatusId::stale &&
                  result.reason == hal::ReasonId::stale_program,
              "foreign executor handle should be rejected") &&
       ok;
  ok = Expect(executor.ResetTemporalState(
                  prepared.handle, hal::ReasonId::temporal_discontinuity) ==
                      hal::StatusId::ready &&
                  runtime.reset_calls == 1,
              "non-temporal discontinuity reset should be harmless") &&
       ok;
  executor.DestroyPrepared(&prepared.handle);

  Fixture fixture;
  ok = Configure(&fixture) && ok;
  hal::HostFrameTokens tokens;
  hal::ReasonId reason = hal::ReasonId::none;
  ok = Expect(fixture.session.BindHostFrame(
                  HostBinding(fixture.program, fixture.input.data(),
                              fixture.output.data(), 1),
                  &tokens, &reason) == hal::StatusId::ready,
              "reconfiguration test should bind") &&
       ok;
  hal::ExecuteRequest request;
  request.invalidation_key = fixture.program.invalidation_key;
  ++request.invalidation_key.generations.configuration;
  request.execution_domain = fixture.program.execution_domain;
  request.capture_sequence = 1;
  request.input = tokens.input;
  result = fixture.session.Execute(request);
  ok = Expect(result.status == hal::StatusId::stale &&
                  result.reason == hal::ReasonId::reconfiguration_required &&
                  fixture.runtime.execute_calls == 0,
              "reconfiguration during execution should reject before "
              "runtime work") &&
       ok;
  return ok;
}

bool TestBoundedTerminalStatesAndPendingShutdown() {
  bool ok = true;
  for (const auto status :
       {RuntimeStatus::timeout, RuntimeStatus::stop_requested,
        RuntimeStatus::cancelled}) {
    Fixture fixture;
    ok = Configure(&fixture) && ok;
    fixture.runtime.execute_status = status;
    const auto result = ExecuteBound(&fixture, 1);
    ok = Expect(result.status != hal::StatusId::executed &&
                    result.status != hal::StatusId::degraded &&
                    !result.output_valid,
                "timeout/stop/cancel must not publish output") &&
         ok;
  }
  {
    FakeRuntime runtime;
    runtime.pending_destroy = true;
    auto options = Options(&runtime);
    open_cuda::OpenCudaPreparedResidentExecutor executor(std::move(options));
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok = Expect(session.Configure(Program(), adapter, &readiness, &reason) ==
                    hal::StatusId::degraded,
                "pending-destroy fixture should prepare") &&
         ok;
    session.Shutdown();

    const auto pending = executor.RequestShutdown();
    bool driver_context_released = false;
    if (pending.driver_context_release_permitted)
      driver_context_released = true;
    ok = Expect(pending.status == RuntimeStatus::completion_pending &&
                    pending.shutdown_requested && !pending.quiescent &&
                    pending.pending_prepared_slots == 1 &&
                    pending.stream_owned && !pending.stream_destroyed &&
                    pending.stream_destructions == 0 &&
                    !pending.driver_context_release_permitted &&
                    !driver_context_released &&
                    executor.ShutdownSnapshot().pending_prepared_slots == 1,
                "pending DMA must be observable and retain the complete "
                "driver/context/stream ownership island") &&
         ok;

    runtime.pending_destroy = false;
    const auto quiescent = executor.RequestShutdown();
    ok = Expect(quiescent.status == RuntimeStatus::ready &&
                    quiescent.quiescent &&
                    quiescent.pending_prepared_slots == 0 &&
                    quiescent.stream_destroyed &&
                    quiescent.stream_destructions == 1 &&
                    quiescent.driver_context_release_permitted &&
                    quiescent.shutdown_attempts > pending.shutdown_attempts,
                "retryable shutdown must become quiescent before allowing "
                "driver/context release") &&
         ok;

    executor.Shutdown();
    const auto repeated = executor.RequestShutdown();
    ok = Expect(repeated.quiescent && repeated.stream_destructions == 1 &&
                    repeated.shutdown_attempts > quiescent.shutdown_attempts &&
                    runtime.destroy_calls >= 1 && runtime.shutdown_calls >= 3,
                "compatible repeated shutdown must destroy the owned stream "
                "exactly once") &&
         ok;
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = TestPreparedGeometryStrengthLaunchesAndBudgets() && ok;
  ok = TestAcceptedBlurAndRemoveThroughSession() && ok;
  ok = TestOptionalFailurePreservesInputWithoutAttribution() && ok;
  ok = TestRequiredFailureDeviceLossAndForbiddenFrameWork() && ok;
  ok = TestModelAndCaptureGeometryMayDiffer() && ok;
  ok = TestEligibilityFailsClosedBeforeRuntimeSetup() && ok;
  ok = TestBudgetEvidenceRollbackAndOldPlanRetention() && ok;
  ok = TestStaleForeignBindingReconfigurationAndReset() && ok;
  ok = TestBoundedTerminalStatesAndPendingShutdown() && ok;
  if (!ok)
    return 1;
  std::cout << "[PASS] prepared Open CUDA resident executor tests\n";
  return 0;
}
