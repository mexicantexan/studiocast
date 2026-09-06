#include "core/open_video/open_cuda_production_hal_owner.h"
#include "core/open_video/video_execution_factory.h"
#include "core/video/video_execution_coordinator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

namespace {

using namespace studiocast;
using OwnerStatus = open_cuda::OpenCudaProductionOwnerStatus;
using RuntimeStatus = open_cuda::PreparedOpenCudaRuntimeStatus;

bool Expect(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "[FAIL] " << message << '\n';
  return condition;
}

struct FakeCudaState {
  cuda::CUcontext current = nullptr;
  cuda::CUcontext primary =
      reinterpret_cast<cuda::CUcontext>(std::uintptr_t{0xC001});
  int context_device = 0;
  int create_calls = 0;
  int destroy_attempts = 0;
  int destroy_successes = 0;
  int primary_releases = 0;
  int fail_create_call = 0;
  bool fail_actual_destroy = false;
  std::array<cuda::CUstream, 16> destroyed{};
  std::size_t destroyed_count = 0;
};

FakeCudaState *g_cuda = nullptr;

cuda::CUresult CudaInit(unsigned int) { return cuda::CUDA_SUCCESS; }
cuda::CUresult CudaModuleLoad(cuda::CUmodule *module, const void *) {
  *module = reinterpret_cast<cuda::CUmodule>(std::uintptr_t{0x2001});
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaModuleGetFunction(cuda::CUfunction *function, cuda::CUmodule,
                                     const char *) {
  *function = reinterpret_cast<cuda::CUfunction>(std::uintptr_t{0x2002});
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaLaunch(cuda::CUfunction, unsigned int, unsigned int,
                          unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, cuda::CUstream, void **,
                          void **) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaAlloc(cuda::CUdeviceptr *pointer, std::size_t *pitch,
                         std::size_t width, std::size_t, unsigned int) {
  *pointer = 0x3000;
  *pitch = width;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaFree(cuda::CUdeviceptr) { return cuda::CUDA_SUCCESS; }
cuda::CUresult CudaCopy(const cuda::CUDA_MEMCPY2D *, cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaMemset(cuda::CUdeviceptr, unsigned char, std::size_t,
                          cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaStreamCreate(cuda::CUstream *stream, unsigned int) {
  if (!g_cuda)
    return cuda::CUDA_ERROR_UNKNOWN;
  ++g_cuda->create_calls;
  if (g_cuda->fail_create_call == g_cuda->create_calls) {
    *stream = nullptr;
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
  }
  *stream = reinterpret_cast<cuda::CUstream>(
      static_cast<std::uintptr_t>(0x7000 + g_cuda->create_calls));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaStreamDestroy(cuda::CUstream stream) {
  if (!g_cuda)
    return cuda::CUDA_ERROR_UNKNOWN;
  ++g_cuda->destroy_attempts;
  const auto identity = reinterpret_cast<std::uintptr_t>(stream);
  if (g_cuda->fail_actual_destroy && identity >= 0x7002)
    return cuda::CUDA_ERROR_INVALID_CONTEXT;
  ++g_cuda->destroy_successes;
  if (g_cuda->destroyed_count < g_cuda->destroyed.size())
    g_cuda->destroyed[g_cuda->destroyed_count++] = stream;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaStreamSynchronize(cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaDeviceCount(int *count) {
  *count = 1;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaDeviceGet(cuda::CUdevice *device, int ordinal) {
  *device = ordinal;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaDeviceUuid(cuda::CUuuid *uuid, cuda::CUdevice device) {
  std::memset(uuid->bytes, 0, sizeof(uuid->bytes));
  uuid->bytes[0] = 0x42;
  uuid->bytes[15] = static_cast<char>(device + 1);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaDevicePci(char *pci, int length, cuda::CUdevice) {
  constexpr char value[] = "0000:01:00.0";
  if (length < static_cast<int>(sizeof(value)))
    return cuda::CUDA_ERROR_UNKNOWN;
  std::memcpy(pci, value, sizeof(value));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaPrimaryRetain(cuda::CUcontext *context, cuda::CUdevice) {
  if (!g_cuda)
    return cuda::CUDA_ERROR_UNKNOWN;
  *context = g_cuda->primary;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaPrimaryRelease(cuda::CUdevice) {
  if (g_cuda)
    ++g_cuda->primary_releases;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaCtxSet(cuda::CUcontext context) {
  if (!g_cuda)
    return cuda::CUDA_ERROR_UNKNOWN;
  g_cuda->current = context;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaCtxGet(cuda::CUcontext *context) {
  if (!g_cuda)
    return cuda::CUDA_ERROR_UNKNOWN;
  *context = g_cuda->current;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaCtxDevice(cuda::CUdevice *device) {
  if (!g_cuda)
    return cuda::CUDA_ERROR_UNKNOWN;
  *device = g_cuda->context_device;
  return cuda::CUDA_SUCCESS;
}

cuda::CudaDriverApi::Functions CudaFunctions() {
  cuda::CudaDriverApi::Functions functions{};
  functions.cuInit = CudaInit;
  functions.cuModuleLoadData = CudaModuleLoad;
  functions.cuModuleGetFunction = CudaModuleGetFunction;
  functions.cuLaunchKernel = CudaLaunch;
  functions.cuMemAllocPitch = CudaAlloc;
  functions.cuMemFree = CudaFree;
  functions.cuMemcpy2DAsync = CudaCopy;
  functions.cuMemsetD8Async = CudaMemset;
  functions.cuStreamCreate = CudaStreamCreate;
  functions.cuStreamDestroy = CudaStreamDestroy;
  functions.cuStreamSynchronize = CudaStreamSynchronize;
  functions.cuDeviceGetCount = CudaDeviceCount;
  functions.cuDeviceGet = CudaDeviceGet;
  functions.cuDeviceGetUuid = CudaDeviceUuid;
  functions.cuDeviceGetPCIBusId = CudaDevicePci;
  functions.cuDevicePrimaryCtxRetain = CudaPrimaryRetain;
  functions.cuDevicePrimaryCtxRelease = CudaPrimaryRelease;
  functions.cuCtxSetCurrent = CudaCtxSet;
  functions.cuCtxGetCurrent = CudaCtxGet;
  functions.cuCtxGetDevice = CudaCtxDevice;
  return functions;
}

bool InitializeFakeDriver(void *context, cuda::CudaDriverApi *driver,
                          std::string *error) noexcept {
  g_cuda = static_cast<FakeCudaState *>(context);
  return driver && driver->InitializeForTesting(CudaFunctions(), error);
}

open_video::ValidatedModelArtifactIdentity Artifact() {
  open_video::ValidatedModelArtifactIdentity artifact{};
  artifact.registry_generation = 3;
  artifact.model_id_sha256[0] = 1;
  artifact.manifest_sha256[0] = 2;
  artifact.artifact_sha256[0] = 3;
  artifact.artifact_set_sha256[0] = 4;
  return artifact;
}

open_video::ModelPack Pack() {
  open_video::ModelPack pack{};
  pack.id = "owner-hermetic-matting";
  pack.display_name = "Owner hermetic matting";
  pack.task = "matting";
  pack.registry_generation = 3;
  pack.registry_manifest_sha256[0] = 2;
  open_video::MattingSpec matting{};
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

open_cuda::OpenCudaProductionBackendRequest
BackendRequest(FakeCudaState *state,
               open_cuda::IPreparedOpenCudaRuntime *runtime = nullptr) {
  open_cuda::OpenCudaProductionBackendRequest request{};
  request.adapter_instance_id = 91;
  request.cuda_device_id = 0;
  request.model_pack = Pack();
  request.model_artifact = Artifact();
  request.configuration_generation = 11;
  request.model_preprocess_generation = 13;
  request.model_artifact_hash = 0xA2;
  request.runtime = runtime;
  request.initialize_driver = InitializeFakeDriver;
  request.initialize_driver_context = state;
  return request;
}

hal::ImageDescriptor Image(hal::HostCoherence coherence,
                           hal::Mutability mutability) {
  hal::ImageDescriptor image{};
  image.width = 16;
  image.height = 8;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = coherence;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes =
      coherence == hal::HostCoherence::not_host_visible ? 16 : 1;
  image.planes[0].row_stride_bytes = 48;
  image.planes[0].size_bytes = 384;
  image.planes[0].width = 16;
  image.planes[0].height = 8;
  image.planes[0].alignment_bytes = image.base_alignment_bytes;
  return image;
}

hal::PreparedProgram Program(const hal::ExecutionDomainId &domain) {
  const auto artifact = Artifact();
  hal::PreparedProgram program{};
  program.program_id = 101;
  program.owner_adapter_instance_id = domain.adapter_instance_id;
  program.execution_domain = domain;
  program.host_input =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  program.host_output =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  program.invalidation_key.generations = {
      .configuration = 11,
      .geometry = 12,
      .model = 13,
      .provider = 14,
      .device = 15,
      .context = domain.logical_context_generation,
  };
  program.invalidation_key.canonical_plan_hash = 0xA1;
  program.invalidation_key.model_artifact_hash = 0xA2;
  program.invalidation_key.execution_domain_hash = 0xA3;
  program.budget = {128 * 1024, 64 * 1024, 4096, 4096, 32, 1};
  auto &stage = program.stages[0];
  stage.stage_id = 1;
  stage.semantic = hal::SemanticEffectId::virtual_background_blur;
  stage.provider = hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  stage.backend = hal::BackendId::cuda;
  stage.execution_domain = domain;
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image =
      Image(hal::HostCoherence::not_host_visible, hal::Mutability::read_only);
  stage.output.kind = hal::StageIoKind::image;
  stage.output.image =
      Image(hal::HostCoherence::not_host_visible, hal::Mutability::read_write);
  stage.model.model_id_hash = artifact.model_id_sha256;
  stage.model.artifact_set_hash = artifact.artifact_set_sha256;
  stage.model.artifact_generation = artifact.registry_generation;
  stage.required = true;
  stage.failure_action = hal::FailureAction::fail_closed;
  stage.circuit_breaker.consecutive_failure_limit = 2;
  stage.circuit_breaker.cooldown_frames = 1;
  stage.circuit_breaker.action = hal::CircuitBreakerAction::quarantine_program;
  stage.scratch_bytes = 4096;
  program.stage_count = 1;
  program.cpu_tails[0].id = hal::CpuTailId::writer_copy;
  program.cpu_tails[0].after_stage_id = 1;
  program.cpu_tails[0].required = true;
  program.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  program.cpu_tails[0].boundary_index = 1;
  program.cpu_tail_count = 1;
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = true;
  program.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  program.strict_provider_selection = true;
  program.requested_execution_domain = domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;
  return program;
}

hal::ReadinessSnapshot Readiness(const hal::PreparedProgram &program) {
  hal::ReadinessSnapshot readiness{};
  readiness.status = hal::StatusId::degraded;
  readiness.reason = hal::ReasonId::cpu_tail_required;
  readiness.requested_backend = hal::BackendId::cuda;
  readiness.selected_backend = hal::BackendId::cuda;
  readiness.provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  readiness.execution_domain = program.execution_domain;
  readiness.ready_stage_mask = 1;
  readiness.available = true;
  readiness.degraded = true;
  return readiness;
}

class PendingRuntime final : public open_cuda::IPreparedOpenCudaRuntime {
public:
  bool pending = true;
  std::uint32_t fail_prepare_count = 0;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  open_cuda::PreparedOpenCudaRuntimeHandle active{};
  std::uint64_t shutdown_calls = 0;
  std::uint64_t stream_destructions = 0;

  open_cuda::PreparedOpenCudaRuntimePrepareResult
  Prepare(const hal::PreparedProgram &program,
          const open_cuda::PreparedOpenCudaProductionConfig &config) noexcept
      override {
    ++prepare_calls;
    open_cuda::PreparedOpenCudaRuntimePrepareResult result{};
    if (fail_prepare_count != 0) {
      --fail_prepare_count;
      result.status = RuntimeStatus::resource_exhausted;
      result.reason = hal::ReasonId::resource_exhausted;
      return result;
    }
    result.status = RuntimeStatus::ready;
    result.reason = hal::ReasonId::none;
    result.handle.handle_id = prepare_calls;
    result.handle.program_id = program.program_id;
    result.handle.invalidation_key = program.invalidation_key;
    result.handle.execution_domain = program.execution_domain;
    result.evidence.status = RuntimeStatus::ready;
    result.evidence.semantic = program.stages[0].semantic;
    result.evidence.provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    result.evidence.execution_domain = program.execution_domain;
    result.evidence.model_artifact = config.model_artifact;
    open_cuda::PreparedOpenCudaVbPlanInput input{};
    input.semantic = program.stages[0].semantic;
    input.frame_width = program.host_input.width;
    input.frame_height = program.host_input.height;
    input.model_width =
        static_cast<std::uint64_t>(config.model_pack->matting->input.width);
    input.model_height =
        static_cast<std::uint64_t>(config.model_pack->matting->input.height);
    input.requested_strength = config.strength;
    input.input_pitch_bytes =
        program.stages[0].input.image.planes[0].row_stride_bytes;
    input.output_pitch_bytes =
        program.stages[0].output.image.planes[0].row_stride_bytes;
    input.model_alpha_capacity_bytes =
        input.model_width * input.model_height * sizeof(float);
    input.input_tensor_capacity_bytes = input.model_alpha_capacity_bytes * 3;
    input.resized_alpha_pitch_bytes = input.frame_width * sizeof(float);
    input.alpha_scratch_pitch_bytes = input.resized_alpha_pitch_bytes;
    input.alpha_feather_pitch_bytes = input.resized_alpha_pitch_bytes;
    input.blur_scratch_pitch_bytes = input.input_pitch_bytes;
    input.blurred_pitch_bytes = input.input_pitch_bytes;
    input.upload_staging_bytes = program.host_input.planes[0].size_bytes;
    input.readback_staging_bytes = program.host_output.planes[0].size_bytes;
    input.available_budget = program.budget;
    result.evidence.geometry_plan =
        open_cuda::PlanPreparedOpenCudaVirtualBackground(input);
    result.evidence.reserved_budget =
        result.evidence.geometry_plan.required_budget;
    result.evidence.strict_cuda_only = true;
    result.evidence.user_stream_bound = true;
    result.evidence.cpu_provider_fallback_disabled = true;
    result.evidence.reusable_resources_prepared = true;
    result.evidence.setup_warmup_complete = true;
    result.evidence.inference_only_frame_path = true;
    result.evidence.device_preprocess = true;
    result.evidence.device_postprocess = true;
    result.evidence.one_upload_one_readback = true;
    result.evidence.no_intermediate_crossings = true;
    result.evidence.single_in_flight = true;
    active = result.handle;
    return result;
  }
  open_cuda::PreparedOpenCudaRuntimeFrameResult ExecuteFrame(
      const open_cuda::PreparedOpenCudaRuntimeHandle &handle,
      const open_cuda::PreparedOpenCudaRuntimeFrame &frame) noexcept override {
    ++execute_calls;
    open_cuda::PreparedOpenCudaRuntimeFrameResult result{};
    if (handle.handle_id == 0 || handle.handle_id != active.handle_id) {
      result.status = RuntimeStatus::foreign_handle;
      result.reason = hal::ReasonId::stale_program;
      return result;
    }
    std::memcpy(frame.output, frame.input,
                std::min(frame.input_span_bytes, frame.output_span_bytes));
    result.status = RuntimeStatus::executed;
    result.reason = hal::ReasonId::none;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    result.transfers.cpu_tail_boundaries = 1;
    result.output_valid = true;
    result.analysis_ran = true;
    result.inference_only_path = true;
    return result;
  }
  RuntimeStatus
  ResetTemporalState(const open_cuda::PreparedOpenCudaRuntimeHandle &,
                     hal::ReasonId) noexcept override {
    return RuntimeStatus::not_prepared;
  }
  RuntimeStatus DestroyPrepared(
      open_cuda::PreparedOpenCudaRuntimeHandle *handle) noexcept override {
    if (pending)
      return RuntimeStatus::completion_pending;
    if (handle)
      *handle = {};
    active = {};
    return RuntimeStatus::ready;
  }
  open_cuda::PreparedOpenCudaShutdownSnapshot
  RequestShutdown() noexcept override {
    ++shutdown_calls;
    if (!pending && stream_destructions == 0)
      ++stream_destructions;
    open_cuda::PreparedOpenCudaShutdownSnapshot snapshot{};
    snapshot.status =
        pending ? RuntimeStatus::completion_pending : RuntimeStatus::ready;
    snapshot.shutdown_attempts = shutdown_calls;
    snapshot.stream_destructions = stream_destructions;
    snapshot.pending_prepared_slots = pending ? 1 : 0;
    snapshot.shutdown_requested = true;
    snapshot.quiescent = !pending;
    snapshot.stream_owned = true;
    snapshot.stream_destroyed = !pending;
    snapshot.driver_context_release_permitted = !pending;
    return snapshot;
  }
  open_cuda::PreparedOpenCudaShutdownSnapshot
  ShutdownSnapshot() const noexcept override {
    open_cuda::PreparedOpenCudaShutdownSnapshot snapshot{};
    snapshot.status =
        pending ? RuntimeStatus::completion_pending : RuntimeStatus::ready;
    snapshot.shutdown_attempts = shutdown_calls;
    snapshot.stream_destructions = stream_destructions;
    snapshot.pending_prepared_slots = pending ? 1 : 0;
    snapshot.shutdown_requested = shutdown_calls != 0;
    snapshot.quiescent = !pending && shutdown_calls != 0;
    snapshot.stream_owned = true;
    snapshot.stream_destroyed = stream_destructions != 0;
    snapshot.driver_context_release_permitted = snapshot.quiescent;
    return snapshot;
  }
  open_cuda::PreparedOpenCudaRuntimeTelemetry
  Telemetry() const noexcept override {
    return {};
  }
};

bool ExactDomainCapabilityAndQuarantineAreTyped() {
  FakeCudaState state{};
  open_cuda::OpenCudaProductionPreparationResult result{};
  auto owner = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      BackendRequest(&state), &result);
  bool ok = Expect(owner && result.bootstrap_domain_ready &&
                       !result.prepared_capability_committed &&
                       result.status == OwnerStatus::domain_ready,
                   "bootstrap should not claim prepared model capability");
  if (!owner)
    return false;

  const auto stream_identity =
      reinterpret_cast<std::uintptr_t>(owner->prepared_stream());
  ok = Expect(state.create_calls == 2 && stream_identity == 0x7002 &&
                  result.execution_domain.queue_or_stream_identity ==
                      stream_identity &&
                  result.context == owner->context_identity() &&
                  result.execution_domain.physical_device.stable_id ==
                      result.context.physical_device.uuid &&
                  result.execution_domain.logical_context_generation ==
                      result.context.logical_generation &&
                  result.execution_domain.allocator_family ==
                      hal::AllocatorFamily::cuda_driver &&
                  result.execution_domain.memory_handle_family ==
                      hal::MemoryHandleFamily::cuda_device_pointer &&
                  result.execution_domain.ownership_model ==
                      hal::OwnershipModel::adapter_exclusive &&
                  result.execution_domain.synchronization_protocol ==
                      hal::SynchronizationProtocol::cuda_stream_ordered,
              "domain must bind exact context, actual non-default stream, "
              "CUDA memory, and exclusive ownership") &&
       ok;

  const auto program = Program(owner->execution_domain());
  const auto readiness = Readiness(program);
  const auto committed = owner->CommitPreparedCapability(program, readiness);
  ok = Expect(committed.status == OwnerStatus::prepared_capability_committed &&
                  committed.prepared_capability_committed &&
                  owner->prepared_capability_committed(),
              "only exact adapter readiness should commit capability") &&
       ok;

  auto foreign_readiness = readiness;
  ++foreign_readiness.execution_domain.adapter_instance_id;
  const auto foreign =
      owner->CommitPreparedCapability(program, foreign_readiness);
  ok = Expect(foreign.status == OwnerStatus::wrong_execution_domain &&
                  owner->prepared_capability_committed(),
              "foreign evidence must retain old committed capability") &&
       ok;

  auto stale_program = program;
  ++stale_program.invalidation_key.generations.context;
  const auto stale = owner->CommitPreparedCapability(stale_program, readiness);
  ok = Expect(stale.status == OwnerStatus::stale_generation &&
                  owner->prepared_capability_committed(),
              "stale generation must retain old capability") &&
       ok;

  auto cpu_readiness = readiness;
  cpu_readiness.provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  const auto cpu = owner->CommitPreparedCapability(program, cpu_readiness);
  ok = Expect(cpu.status == OwnerStatus::capability_evidence_incomplete &&
                  owner->prepared_capability_committed(),
              "CPU evidence must never promote Open CUDA capability") &&
       ok;

  auto exhausted_readiness = readiness;
  exhausted_readiness.status = hal::StatusId::resource_exhausted;
  exhausted_readiness.reason = hal::ReasonId::resource_exhausted;
  exhausted_readiness.available = false;
  const auto exhausted =
      owner->CommitPreparedCapability(program, exhausted_readiness);
  ok = Expect(exhausted.status == OwnerStatus::resource_exhausted &&
                  exhausted.hal_status == hal::StatusId::resource_exhausted &&
                  exhausted.reason == hal::ReasonId::resource_exhausted &&
                  owner->prepared_capability_committed(),
              "resource exhaustion must remain typed and retain old "
              "capability") &&
       ok;

  auto foreign_program = program;
  ++foreign_program.owner_adapter_instance_id;
  const auto foreign_adapter =
      owner->CommitPreparedCapability(foreign_program, readiness);
  if (foreign_adapter.status != OwnerStatus::foreign_adapter) {
    std::cerr << "[DETAIL] foreign status="
              << static_cast<unsigned>(foreign_adapter.status)
              << " reason=" << static_cast<unsigned>(foreign_adapter.reason)
              << '\n';
  }
  ok = Expect(foreign_adapter.status == OwnerStatus::foreign_adapter &&
                  owner->prepared_capability_committed(),
              "foreign adapter ownership must fail closed") &&
       ok;

  auto malformed_program = program;
  malformed_program.execution_domain.queue_or_stream_identity = 0;
  malformed_program.requested_execution_domain =
      malformed_program.execution_domain;
  malformed_program.stages[0].execution_domain =
      malformed_program.execution_domain;
  const auto malformed_domain =
      owner->CommitPreparedCapability(malformed_program, readiness);
  if (malformed_domain.reason != hal::ReasonId::malformed_execution_domain) {
    std::cerr << "[DETAIL] malformed status="
              << static_cast<unsigned>(malformed_domain.status)
              << " reason=" << static_cast<unsigned>(malformed_domain.reason)
              << '\n';
  }
  ok = Expect(
           malformed_domain.status ==
                   OwnerStatus::capability_evidence_incomplete &&
               malformed_domain.reason ==
                   hal::ReasonId::malformed_execution_domain &&
               owner->prepared_capability_committed(),
           "malformed execution domain must fail before capability mutation") &&
       ok;

  owner->ObserveRuntimeStatus(hal::StatusId::quarantined,
                              hal::ReasonId::device_lost);
  const auto snapshot = owner->Snapshot();
  ok = Expect(snapshot.quarantined && snapshot.device_lost &&
                  !snapshot.prepared_capability.committed &&
                  !owner->prepared_capability_committed(),
              "device loss must quarantine and revoke capability") &&
       ok;

  const auto shutdown = owner->RequestShutdown();
  ok = Expect(
           shutdown.quiescent && shutdown.driver_context_release_permitted &&
               state.destroy_successes == 2 && state.destroyed_count == 2 &&
               reinterpret_cast<std::uintptr_t>(state.destroyed[0]) == 0x7001 &&
               reinterpret_cast<std::uintptr_t>(state.destroyed[1]) == 0x7002,
           "validation and owned streams must each be destroyed once") &&
       ok;
  return ok;
}

bool CandidateFailuresRetainOldIslandAndRollbackExactStream() {
  {
    FakeCudaState malformed_state{};
    auto malformed_request = BackendRequest(&malformed_state);
    malformed_request.adapter_instance_id = 0;
    open_cuda::OpenCudaProductionPreparationResult malformed_result{};
    auto malformed = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
        malformed_request, &malformed_result);
    if (!Expect(!malformed &&
                    malformed_result.status == OwnerStatus::malformed_request &&
                    malformed_state.create_calls == 0,
                "malformed bootstrap request must fail before CUDA work")) {
      return false;
    }
  }

  FakeCudaState old_state{};
  open_cuda::OpenCudaProductionPreparationResult result{};
  auto old_owner = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      BackendRequest(&old_state), &result);
  bool ok = Expect(old_owner && old_owner->bootstrap_domain_ready(),
                   "old valid island should bootstrap");
  if (!old_owner)
    return false;

  FakeCudaState mismatch_state{};
  auto mismatch_request = BackendRequest(&mismatch_state);
  mismatch_request.require_expected_context = true;
  mismatch_request.expected_context = old_owner->context_identity();
  auto mismatch = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      mismatch_request, &result);
  ok = Expect(!mismatch && result.status == OwnerStatus::context_mismatch &&
                  result.reason == hal::ReasonId::wrong_generation &&
                  mismatch_state.create_calls == 2 &&
                  mismatch_state.destroy_successes == 2 &&
                  old_owner->bootstrap_domain_ready() &&
                  old_owner->driver() != nullptr &&
                  old_owner->adapter() != nullptr,
              "failed candidate must roll back and leave old owner intact") &&
       ok;

  FakeCudaState create_failure_state{};
  create_failure_state.fail_create_call = 2;
  auto create_failure = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      BackendRequest(&create_failure_state), &result);
  ok = Expect(!create_failure &&
                  result.status == OwnerStatus::stream_create_failed &&
                  create_failure_state.create_calls == 2 &&
                  create_failure_state.destroy_successes == 1 &&
                  old_owner->bootstrap_domain_ready(),
              "actual stream-create failure must retain old owner") &&
       ok;

  FakeCudaState cleanup_state{};
  cleanup_state.fail_actual_destroy = true;
  auto cleanup_request = BackendRequest(&cleanup_state);
  cleanup_request.require_expected_context = true;
  cleanup_request.expected_context = old_owner->context_identity();
  auto cleanup_owner = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      cleanup_request, &result);
  ok = Expect(cleanup_owner && result.retained_cleanup_island &&
                  result.status == OwnerStatus::stream_destroy_failed &&
                  cleanup_owner->driver() != nullptr &&
                  cleanup_owner->prepared_stream() != nullptr &&
                  !cleanup_owner->ShutdownSnapshot()
                       .driver_context_release_permitted,
              "failed rollback destruction must retain the partial island") &&
       ok;
  if (cleanup_owner) {
    cleanup_state.fail_actual_destroy = false;
    const auto cleaned = cleanup_owner->RequestShutdown();
    ok = Expect(cleaned.quiescent && cleaned.driver_context_release_permitted &&
                    cleaned.owner_stream_destructions == 1 &&
                    cleanup_state.destroy_successes == 2,
                "retry must destroy retained stream before context release") &&
         ok;
  }

  g_cuda = &old_state;
  (void)old_owner->RequestShutdown();
  return ok;
}

bool ShutdownRetainsCompleteIslandUntilQuiescent() {
  FakeCudaState state{};
  PendingRuntime runtime{};
  open_cuda::OpenCudaProductionPreparationResult result{};
  auto owner = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      BackendRequest(&state, &runtime), &result);
  bool ok = Expect(owner && owner->bootstrap_domain_ready(),
                   "pending fixture should bootstrap");
  if (!owner)
    return false;

  const auto pending = owner->RequestShutdown();
  ok = Expect(pending.status == OwnerStatus::shutdown_pending &&
                  !pending.quiescent &&
                  !pending.driver_context_release_permitted &&
                  pending.executor.pending_prepared_slots == 1 &&
                  owner->driver() != nullptr && owner->executor() != nullptr &&
                  owner->adapter() != nullptr &&
                  owner->prepared_stream() != nullptr &&
                  state.primary_releases == 0,
              "pending work must retain the complete island") &&
       ok;

  const auto repeated_pending = owner->RequestShutdown();
  ok = Expect(!repeated_pending.driver_context_release_permitted &&
                  runtime.stream_destructions == 0 &&
                  state.primary_releases == 0,
              "repeated pending shutdown must not release resources") &&
       ok;

  runtime.pending = false;
  const auto quiescent = owner->RequestShutdown();
  ok = Expect(
           quiescent.quiescent && quiescent.driver_context_release_permitted &&
               quiescent.executor.stream_destructions == 1 &&
               runtime.stream_destructions == 1 && owner->driver() == nullptr &&
               owner->executor() == nullptr && owner->adapter() == nullptr &&
               owner->prepared_stream() == nullptr &&
               state.primary_releases == 1,
           "quiescent retry must release the island in order") &&
       ok;

  const auto repeated = owner->RequestShutdown();
  ok = Expect(repeated.quiescent && runtime.stream_destructions == 1 &&
                  state.primary_releases == 1,
              "completed shutdown must not double destroy") &&
       ok;
  return ok;
}

bool ProductionStreamDestroyFailureIsRetryable() {
  FakeCudaState state{};
  open_cuda::OpenCudaProductionPreparationResult result{};
  auto owner = open_cuda::OpenCudaProductionHalOwner::CreateBackend(
      BackendRequest(&state), &result);
  bool ok = Expect(owner && owner->bootstrap_domain_ready(),
                   "destroy-failure fixture should bootstrap");
  if (!owner)
    return false;

  state.fail_actual_destroy = true;
  const auto pending = owner->RequestShutdown();
  ok = Expect(pending.status == OwnerStatus::shutdown_pending &&
                  !pending.driver_context_release_permitted &&
                  pending.executor.stream_owned &&
                  !pending.executor.stream_destroyed &&
                  owner->driver() != nullptr && owner->executor() != nullptr &&
                  state.primary_releases == 0,
              "stream destroy failure must retain the production island") &&
       ok;

  state.fail_actual_destroy = false;
  const auto complete = owner->RequestShutdown();
  ok = Expect(complete.quiescent && complete.driver_context_release_permitted &&
                  complete.executor.stream_destructions == 1 &&
                  state.primary_releases == 1,
              "production destroy retry must succeed exactly once") &&
       ok;
  const int successes = state.destroy_successes;
  (void)owner->RequestShutdown();
  ok = Expect(state.destroy_successes == successes &&
                  state.primary_releases == 1,
              "post-quiescence retry must be idempotent") &&
       ok;
  return ok;
}

video::VideoExecutionPlanRequest
VideoRequest(const hal::ExecutionDomainId &domain) {
  const auto program = Program(domain);
  video::VideoExecutionPlanRequest request{};
  request.plan_id = program.program_id;
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
  request.intent.stages[0].intent.stage_id = program.stages[0].stage_id;
  request.intent.stages[0].intent.semantic = program.stages[0].semantic;
  request.intent.stages[0].intent.required = true;
  request.intent.stages[0].intent.failure_action =
      hal::FailureAction::fail_closed;
  request.intent.stages[0].model = program.stages[0].model;
  request.intent.stage_count = 1;
  request.host_input = program.host_input;
  request.host_output = program.host_output;
  request.invalidation_key = program.invalidation_key;
  request.temporal.frame_drop_behavior =
      hal::FrameDropBehavior::latest_frame_wins;
  request.budget = program.budget;
  request.cpu_tails = program.cpu_tails;
  request.cpu_tail_count = program.cpu_tail_count;
  request.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  request.migration_policy =
      video::VideoExecutionMigrationPolicy::prepared_only;
  return request;
}

open_cuda::OpenCudaProductionVideoExecutionFactoryRequest
ProductionFactoryRequest(FakeCudaState *state, PendingRuntime *runtime,
                         hal::SemanticEffectId semantic =
                             hal::SemanticEffectId::virtual_background_blur) {
  auto backend = BackendRequest(state, runtime);
  if (semantic == hal::SemanticEffectId::virtual_background_replace) {
    backend.replacement_width = 2;
    backend.replacement_height = 2;
    backend.replacement_rgb = {
        255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
    };
  }
  // The neutral setup lease binds its model generation to the exact validated
  // artifact registry generation.
  backend.model_preprocess_generation =
      backend.model_artifact.registry_generation;
  open_cuda::OpenCudaProductionVideoExecutionFactoryRequest setup{};
  setup.factory_id = 7011;
  setup.selection_priority = 5;
  setup.stage_count = 1;
  setup.semantics[0] = semantic;
  setup.models[0].model_id_hash = backend.model_artifact.model_id_sha256;
  setup.models[0].artifact_set_hash =
      backend.model_artifact.artifact_set_sha256;
  setup.models[0].artifact_generation =
      backend.model_artifact.registry_generation;
  setup.generations = {
      11, 12, backend.model_artifact.registry_generation, 14, 15, 0,
  };
  setup.reserved_budget = {128 * 1024, 64 * 1024, 4096, 4096, 32, 1};
  setup.backend = std::move(backend);
  setup.model_artifact_validated = true;
  setup.provider_runtime_validated = true;
  setup.cuda_iobinding_validated = true;
  setup.user_stream_bound = true;
  setup.reusable_resources_supported = true;
  setup.device_preprocess_supported = true;
  setup.device_postprocess_supported = true;
  setup.stop_aware_bounded_wait = true;
  setup.allocation_free_execute = true;
  return setup;
}

bool ProductionNewSemanticContractsAreExact() {
  bool ok = true;
  for (const auto semantic : {hal::SemanticEffectId::virtual_background_replace,
                              hal::SemanticEffectId::virtual_key_light}) {
    FakeCudaState state{};
    PendingRuntime runtime{};
    runtime.pending = false;
    auto setup = ProductionFactoryRequest(&state, &runtime, semantic);
    open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
    auto factory = open_cuda::OpenCudaVideoExecutionFactory::CreateProduction(
        setup, &bootstrap);
    ok = Expect(factory && bootstrap.status == hal::StatusId::ready &&
                    bootstrap.exact_candidate_ready &&
                    factory->adapter_registration()
                            .semantic_capabilities[0]
                            .semantic == semantic &&
                    factory->adapter_registration()
                        .semantic_capabilities[0]
                        .production_eligible,
                "replace/key-light production capability must be exact and "
                "selectable") &&
         ok;
    factory.reset();
    ok =
        Expect(state.destroy_successes == 1 && runtime.stream_destructions == 1,
               "new semantic owner must retain normal quiescent teardown") &&
        ok;
  }

  {
    FakeCudaState state{};
    PendingRuntime runtime{};
    auto invalid = ProductionFactoryRequest(
        &state, &runtime, hal::SemanticEffectId::virtual_background_replace);
    invalid.backend.replacement_rgb.pop_back();
    open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
    const auto factory =
        open_cuda::OpenCudaVideoExecutionFactory::CreateProduction(invalid,
                                                                   &bootstrap);
    ok = Expect(!factory && bootstrap.status == hal::StatusId::blocked &&
                    bootstrap.reason ==
                        hal::ReasonId::capability_evidence_incomplete &&
                    state.create_calls == 0,
                "malformed replacement resources must fail before CUDA "
                "ownership") &&
         ok;
  }
  {
    FakeCudaState state{};
    PendingRuntime runtime{};
    auto invalid = ProductionFactoryRequest(
        &state, &runtime, hal::SemanticEffectId::virtual_key_light);
    invalid.backend.key_light_intensity = 101;
    open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
    const auto factory =
        open_cuda::OpenCudaVideoExecutionFactory::CreateProduction(invalid,
                                                                   &bootstrap);
    ok = Expect(!factory && bootstrap.status == hal::StatusId::blocked &&
                    bootstrap.reason ==
                        hal::ReasonId::capability_evidence_incomplete &&
                    state.create_calls == 0,
                "out-of-contract key-light parameters must fail before CUDA "
                "ownership") &&
         ok;
  }
  return ok;
}

bool ProductionSetupLeaseIsExactStableAndIdempotent() {
  FakeCudaState state{};
  PendingRuntime runtime{};
  runtime.pending = false;
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence{};
  open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
  hal::ReasonId reason = hal::ReasonId::none;
  const auto setup = ProductionFactoryRequest(&state, &runtime);
  const auto status =
      open_cuda::OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
          setup, &lease, &bootstrap, &evidence, &reason);
  bool ok = Expect(status == hal::StatusId::ready && lease.live() &&
                       bootstrap.exact_candidate_ready &&
                       bootstrap.status == hal::StatusId::ready &&
                       evidence.status == hal::StatusId::ready &&
                       reason == hal::ReasonId::none,
                   "exact Open CUDA setup should acquire a typed live lease");
  if (!lease.live())
    return false;

  const auto &snapshot = lease.snapshot();
  auto *resident = snapshot.adapter.resident_adapter;
  auto *factory = snapshot.factory.factory;
  ok = Expect(
           snapshot.contract_version ==
                   video::kProductionVideoSetupLeaseContractVersion &&
               snapshot.backend == hal::BackendId::cuda &&
               snapshot.execution_domain == bootstrap.owner.execution_domain &&
               snapshot.generations.context ==
                   snapshot.execution_domain.logical_context_generation &&
               snapshot.generations.model ==
                   setup.models[0].artifact_generation &&
               snapshot.providers[0] ==
                   hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
               snapshot.provider_count == 1 && snapshot.artifact_count == 1 &&
               snapshot.artifacts[0].semantic == setup.semantics[0] &&
               snapshot.artifacts[0].artifact.model_id_hash ==
                   setup.models[0].model_id_hash &&
               snapshot.reserved_budget.persistent_bytes ==
                   setup.reserved_budget.persistent_bytes &&
               snapshot.reserved_budget.scratch_bytes ==
                   setup.reserved_budget.scratch_bytes &&
               snapshot.reserved_budget.upload_staging_bytes ==
                   setup.reserved_budget.upload_staging_bytes &&
               snapshot.reserved_budget.readback_staging_bytes ==
                   setup.reserved_budget.readback_staging_bytes &&
               snapshot.reserved_budget.resource_count ==
                   setup.reserved_budget.resource_count &&
               snapshot.reserved_budget.in_flight_slots == 1 &&
               resident != nullptr && factory != nullptr &&
               snapshot.adapter.adapter_instance_id ==
                   snapshot.factory.adapter_instance_id,
           "lease snapshot must freeze exact domain, provider, model, budget, "
           "adapter, and factory evidence") &&
       ok;

  lease.Shutdown();
  lease.Shutdown();
  const auto &stopped = lease.snapshot();
  auto *typed_factory =
      static_cast<open_cuda::OpenCudaVideoExecutionFactory *>(factory);
  const auto stopped_adapter = typed_factory->adapter_registration();
  const auto stopped_factory = typed_factory->Registration();
  ok = Expect(!lease.live() &&
                  stopped.state ==
                      video::ProductionVideoSetupLeaseState::stopped &&
                  stopped.adapter.resident_adapter == resident &&
                  stopped.factory.factory == factory &&
                  stopped_adapter.resident_adapter == resident &&
                  stopped_factory.factory == factory &&
                  resident->adapter_instance_id() ==
                      stopped.adapter.adapter_instance_id &&
                  resident->backend_id() == hal::BackendId::cuda &&
                  state.destroy_successes == 1 &&
                  runtime.stream_destructions == 1,
              "double shutdown must retain published addresses and tear down "
              "native streams exactly once") &&
       ok;
  return ok;
}

bool ProductionSetupLeaseMismatchRollsBackAndRetainsOldLease() {
  FakeCudaState old_state{};
  PendingRuntime old_runtime{};
  old_runtime.pending = false;
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence{};
  open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
  hal::ReasonId reason = hal::ReasonId::none;
  const auto valid = ProductionFactoryRequest(&old_state, &old_runtime);
  bool ok = Expect(
      open_cuda::OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
          valid, &lease, &bootstrap, &evidence, &reason) ==
              hal::StatusId::ready &&
          lease.live(),
      "rollback fixture should start with one live lease");
  if (!lease.live())
    return false;
  const auto old_domain = lease.snapshot().execution_domain;
  auto *old_adapter = lease.snapshot().adapter.resident_adapter;

  auto wrong_model = valid;
  wrong_model.models[0].model_id_hash[0] ^= 0xff;
  evidence = {};
  bootstrap = {};
  reason = hal::ReasonId::none;
  const auto model_status =
      open_cuda::OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
          wrong_model, &lease, &bootstrap, &evidence, &reason);
  ok = Expect(model_status == hal::StatusId::blocked &&
                  reason == hal::ReasonId::capability_evidence_incomplete &&
                  lease.live() &&
                  lease.snapshot().execution_domain == old_domain &&
                  lease.snapshot().adapter.resident_adapter == old_adapter &&
                  evidence.previous_lease_retained &&
                  !evidence.candidate_owner_rolled_back &&
                  old_state.destroy_successes == 1,
              "model mismatch must fail before CUDA work and retain the old "
              "typed lease") &&
       ok;

  FakeCudaState stale_state{};
  PendingRuntime stale_runtime{};
  stale_runtime.pending = false;
  auto stale = ProductionFactoryRequest(&stale_state, &stale_runtime);
  ++stale.generations.model;
  stale.backend.model_preprocess_generation = stale.generations.model;
  evidence = {};
  bootstrap = {};
  reason = hal::ReasonId::none;
  const auto stale_status =
      open_cuda::OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
          stale, &lease, &bootstrap, &evidence, &reason);
  ok = Expect(stale_status == hal::StatusId::stale &&
                  reason == hal::ReasonId::wrong_generation &&
                  evidence.candidate_owner_rolled_back &&
                  evidence.previous_lease_retained && lease.live() &&
                  lease.snapshot().execution_domain == old_domain &&
                  lease.snapshot().adapter.resident_adapter == old_adapter &&
                  stale_state.destroy_successes == 1 &&
                  stale_runtime.stream_destructions == 1,
              "model-generation mismatch must roll back the complete new "
              "island and retain the old lease") &&
       ok;

  g_cuda = &old_state;
  lease.Shutdown();
  ok = Expect(old_state.destroy_successes == 1 &&
                  old_runtime.stream_destructions == 1,
              "retained old lease must still tear down exactly once") &&
       ok;
  return ok;
}

bool ProductionSetupLeaseShutdownRetriesPendingThenComplete() {
  FakeCudaState state{};
  PendingRuntime runtime{};
  runtime.pending = true;
  const auto setup = ProductionFactoryRequest(&state, &runtime);
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence{};
  open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Expect(
      open_cuda::OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
          setup, &lease, &bootstrap, &evidence, &reason) ==
              hal::StatusId::ready &&
          lease.live(),
      "pending shutdown fixture should acquire one live lease");
  if (!lease.live())
    return false;

  auto *factory = static_cast<open_cuda::OpenCudaVideoExecutionFactory *>(
      lease.snapshot().factory.factory);
  auto *resident = lease.snapshot().adapter.resident_adapter;
  const auto pending = lease.Shutdown();
  ok =
      Expect(pending.pending && !pending.quiescent &&
                 !pending.driver_context_release_permitted &&
                 pending.shutdown_attempts == 1 && lease.shutdown_pending() &&
                 lease.snapshot().state ==
                     video::ProductionVideoSetupLeaseState::shutdown_pending &&
                 !factory->production_owner_retired() &&
                 !factory->production_shutdown_snapshot()
                      .driver_context_release_permitted &&
                 runtime.shutdown_calls == 2 &&
                 runtime.stream_destructions == 0 &&
                 state.primary_releases == 0 &&
                 resident == lease.snapshot().adapter.resident_adapter &&
                 resident->adapter_instance_id() ==
                     lease.snapshot().adapter.adapter_instance_id,
             "first lease shutdown call must remain truthfully pending without "
             "retiring the facade or releasing the CUDA context") &&
      ok;

  runtime.pending = false;
  const auto complete = lease.Shutdown();
  ok = Expect(
           !complete.pending && complete.quiescent &&
               complete.driver_context_release_permitted &&
               complete.status == hal::StatusId::stopped &&
               complete.reason == hal::ReasonId::shutdown_requested &&
               complete.shutdown_attempts == 2 && !lease.shutdown_pending() &&
               lease.snapshot().state ==
                   video::ProductionVideoSetupLeaseState::stopped &&
               factory->production_owner_retired() &&
               factory->production_shutdown_snapshot()
                   .driver_context_release_permitted &&
               runtime.shutdown_calls == 4 &&
               runtime.stream_destructions == 1 && state.primary_releases == 1,
           "second explicit lease shutdown call must retire only after native "
           "quiescence and context-release proof") &&
       ok;
  return ok;
}

bool ProductionSetupLeasePermanentPendingIsTruthfulAndFailSafe() {
  FakeCudaState state{};
  PendingRuntime runtime{};
  runtime.pending = true;
  const auto setup = ProductionFactoryRequest(&state, &runtime);
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence{};
  open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Expect(
      open_cuda::OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
          setup, &lease, &bootstrap, &evidence, &reason) ==
              hal::StatusId::ready &&
          lease.live(),
      "permanent pending fixture should acquire one live lease");
  if (!lease.live())
    return false;

  auto *factory = static_cast<open_cuda::OpenCudaVideoExecutionFactory *>(
      lease.snapshot().factory.factory);
  for (std::uint64_t attempt = 1; attempt <= 3; ++attempt) {
    const auto pending = lease.Shutdown();
    ok =
        Expect(
            pending.pending && !pending.quiescent &&
                !pending.driver_context_release_permitted &&
                pending.shutdown_attempts == attempt &&
                lease.shutdown_pending() &&
                lease.snapshot().state ==
                    video::ProductionVideoSetupLeaseState::shutdown_pending &&
                !factory->production_owner_retired() &&
                runtime.shutdown_calls == attempt + 1 &&
                runtime.stream_destructions == 0 && state.primary_releases == 0,
            "each permanently-pending retry must perform one bounded native "
            "attempt and retain the complete unsafe island") &&
        ok;
  }

  // Make the hermetic fake quiescent only for explicit cleanup so ASan can
  // prove the retained fail-safe island is eventually released exactly once.
  runtime.pending = false;
  const auto cleanup = lease.Shutdown();
  ok = Expect(cleanup.driver_context_release_permitted && cleanup.quiescent &&
                  !cleanup.pending && cleanup.shutdown_attempts == 4 &&
                  factory->production_owner_retired() &&
                  runtime.shutdown_calls == 6 &&
                  runtime.stream_destructions == 1 &&
                  state.primary_releases == 1,
              "an explicitly retried quiescent island must complete without "
              "double release after persistent pending evidence") &&
       ok;
  return ok;
}

bool ProductionFactoryReprepareAndRollbackAreLifetimeSafe() {
  FakeCudaState state{};
  PendingRuntime runtime{};
  runtime.pending = false;
  auto backend = BackendRequest(&state, &runtime);
  open_cuda::OpenCudaProductionVideoExecutionFactoryRequest setup{};
  setup.factory_id = 7009;
  setup.selection_priority = 5;
  setup.stage_count = 1;
  setup.semantics[0] = hal::SemanticEffectId::virtual_background_blur;
  setup.models[0].model_id_hash = backend.model_artifact.model_id_sha256;
  setup.models[0].artifact_set_hash =
      backend.model_artifact.artifact_set_sha256;
  setup.models[0].artifact_generation =
      backend.model_artifact.registry_generation;
  setup.generations = {11, 12, 13, 14, 15, 0};
  setup.reserved_budget = {128 * 1024, 64 * 1024, 4096, 4096, 32, 1};
  setup.backend = backend;
  setup.model_artifact_validated = true;
  setup.provider_runtime_validated = true;
  setup.cuda_iobinding_validated = true;
  setup.user_stream_bound = true;
  setup.reusable_resources_supported = true;
  setup.device_preprocess_supported = true;
  setup.device_postprocess_supported = true;
  setup.stop_aware_bounded_wait = true;
  setup.allocation_free_execute = true;

  open_cuda::OpenCudaProductionVideoExecutionBootstrapResult bootstrap{};
  auto factory = open_cuda::OpenCudaVideoExecutionFactory::CreateProduction(
      setup, &bootstrap);
  bool ok = Expect(factory && bootstrap.exact_candidate_ready &&
                       bootstrap.status == hal::StatusId::ready,
                   "production factory bootstrap should retain exact owner");
  if (!factory)
    return false;
  const auto adapter = factory->adapter_registration();
  const auto factory_registration = factory->Registration();
  ok = Expect(adapter.resident_adapter != nullptr &&
                  adapter.execution_domain ==
                      factory_registration.execution_domain &&
                  adapter.adapter_instance_id ==
                      factory_registration.adapter_instance_id,
              "factory and registry must expose the same exact adapter") &&
       ok;

  hal::StaticAdapterRegistry registry;
  hal::ReasonId reason = hal::ReasonId::none;
  ok = Expect(registry.Register(adapter, &reason) == hal::StatusId::ready,
              "exact production adapter registration should be selectable") &&
       ok;
  video::VideoExecutionCoordinator coordinator(registry);
  ok = Expect(coordinator.RegisterFactory(factory_registration, &reason) ==
                  hal::StatusId::ready,
              "exact production factory registration should be accepted") &&
       ok;
  auto request = VideoRequest(adapter.execution_domain);
  runtime.fail_prepare_count = 1;
  video::VideoExecutionPrepareEvidence evidence{};
  ok = Expect(coordinator.Prepare(request, &evidence, &reason) ==
                      hal::StatusId::resource_exhausted &&
                  runtime.prepare_calls == 1 &&
                  factory->adapter_registration().resident_adapter ==
                      adapter.resident_adapter,
              "failed prepared slot must roll back without retiring registry "
              "owner") &&
       ok;
  const auto retry_status = coordinator.Prepare(request, &evidence, &reason);
  if (retry_status != hal::StatusId::ready &&
      retry_status != hal::StatusId::degraded)
    std::cerr << "[DETAIL] retry status=" << static_cast<unsigned>(retry_status)
              << " reason=" << static_cast<unsigned>(reason)
              << " calls=" << runtime.prepare_calls << '\n';
  ok = Expect((retry_status == hal::StatusId::ready ||
               retry_status == hal::StatusId::degraded) &&
                  runtime.prepare_calls == 2,
              "same exact factory must retry after failed preparation") &&
       ok;

  std::array<std::byte, 384> input{};
  std::array<std::byte, 384> output{};
  input.fill(std::byte{0x5a});
  hal::HostFrameBinding binding{};
  binding.input_descriptor = request.host_input;
  binding.output_descriptor = request.host_output;
  binding.input_planes[0].read_data = input.data();
  binding.input_planes[0].span_bytes = input.size();
  binding.input_planes[0].row_stride_bytes = 48;
  binding.output_planes[0].write_data = output.data();
  binding.output_planes[0].span_bytes = output.size();
  binding.output_planes[0].row_stride_bytes = 48;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = 1;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  const auto frame = coordinator.Execute(binding);
  ok = Expect(frame.status == hal::StatusId::executed && frame.output_ready &&
                  frame.transfers.uploads == 1 &&
                  frame.transfers.final_readbacks == 1 &&
                  frame.transfers.intermediate_uploads == 0 &&
                  frame.transfers.intermediate_readbacks == 0 &&
                  coordinator.CommitOutput(frame.completion, true, &reason) ==
                      hal::StatusId::ready &&
                  coordinator.Snapshot().latest_committed_frame.valid,
              "production execution should preserve transfer and commit "
              "attribution") &&
       ok;

  coordinator.Invalidate(hal::ReasonId::reconfiguration_required);
  const auto reprepare_status =
      coordinator.Prepare(request, &evidence, &reason);
  if (reprepare_status != hal::StatusId::ready &&
      reprepare_status != hal::StatusId::degraded)
    std::cerr << "[DETAIL] reprepare status="
              << static_cast<unsigned>(reprepare_status)
              << " reason=" << static_cast<unsigned>(reason)
              << " calls=" << runtime.prepare_calls << '\n';
  ok = Expect((reprepare_status == hal::StatusId::ready ||
               reprepare_status == hal::StatusId::degraded) &&
                  runtime.prepare_calls == 3 &&
                  factory->adapter_registration().resident_adapter ==
                      adapter.resident_adapter,
              "invalidate/reprepare must keep registry adapter alive") &&
       ok;
  coordinator.Shutdown();
  factory.reset();
  if (state.destroy_successes != 1 || runtime.stream_destructions != 1)
    std::cerr << "[DETAIL] stream destroy successes=" << state.destroy_successes
              << " attempts=" << state.destroy_attempts << '\n';
  ok =
      Expect(state.destroy_successes == 1 && runtime.stream_destructions == 1,
             "factory retirement should destroy validation and injected "
             "runtime-owned streams exactly once after coordinator shutdown") &&
      ok;
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = ExactDomainCapabilityAndQuarantineAreTyped() && ok;
  ok = CandidateFailuresRetainOldIslandAndRollbackExactStream() && ok;
  ok = ShutdownRetainsCompleteIslandUntilQuiescent() && ok;
  ok = ProductionStreamDestroyFailureIsRetryable() && ok;
  ok = ProductionNewSemanticContractsAreExact() && ok;
  ok = ProductionSetupLeaseIsExactStableAndIdempotent() && ok;
  ok = ProductionSetupLeaseMismatchRollsBackAndRetainsOldLease() && ok;
  ok = ProductionSetupLeaseShutdownRetriesPendingThenComplete() && ok;
  ok = ProductionSetupLeasePermanentPendingIsTruthfulAndFailSafe() && ok;
  ok = ProductionFactoryReprepareAndRollbackAreLifetimeSafe() && ok;
  g_cuda = nullptr;
  if (!ok)
    return EXIT_FAILURE;
  std::cout << "[PASS] Open CUDA production HAL owner tests\n";
  return EXIT_SUCCESS;
}
