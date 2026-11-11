#include "core/maxine/production_hal_owner.h"
#include "core/maxine/production_video_execution_bootstrap.h"
#include "core/video/video_execution_coordinator.h"
#include "maxine_effect_runtime_fake_control.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>
#include <utility>

namespace {

using namespace studiocast;
using namespace studiocast::maxine;

std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_stream_destroy_calls{0};
thread_local bool g_count_allocations = false;

void *Allocate(std::size_t size) {
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    if (g_count_allocations)
      g_allocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
  }
  throw std::bad_alloc();
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << __func__ << ':' << __LINE__                                 \
                << ": CHECK failed: " << #condition << '\n';                   \
      return false;                                                            \
    }                                                                          \
  } while (false)

struct FakeCudaState {
  cuda::CUcontext current = nullptr;
  int context_device = 0;
};

FakeCudaState *g_cuda = nullptr;

cuda::CUresult CudaInit(unsigned int) { return cuda::CUDA_SUCCESS; }
cuda::CUresult CudaModuleLoad(cuda::CUmodule *module, const void *) {
  *module = reinterpret_cast<cuda::CUmodule>(0x21);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaModuleGetFunction(cuda::CUfunction *function, cuda::CUmodule,
                                     const char *) {
  *function = reinterpret_cast<cuda::CUfunction>(0x22);
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
  *pointer = 0x1000;
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
  *stream = reinterpret_cast<cuda::CUstream>(0x30);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaStreamDestroy(cuda::CUstream) {
  g_stream_destroy_calls.fetch_add(1, std::memory_order_relaxed);
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
    return 1;
  std::memcpy(pci, value, sizeof(value));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaPrimaryRetain(cuda::CUcontext *context,
                                 cuda::CUdevice device) {
  *context = reinterpret_cast<cuda::CUcontext>(
      static_cast<std::uintptr_t>(0x100 + device));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaPrimaryRelease(cuda::CUdevice) { return cuda::CUDA_SUCCESS; }
cuda::CUresult CudaCtxSet(cuda::CUcontext context) {
  if (g_cuda)
    g_cuda->current = context;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaCtxGet(cuda::CUcontext *context) {
  *context = g_cuda ? g_cuda->current : nullptr;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CudaCtxDevice(cuda::CUdevice *device) {
  *device = g_cuda ? g_cuda->context_device : -1;
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

class FakeDeviceOps final : public IResidentDeviceOps {
public:
  bool Initialize(cuda::CudaDriverApi *, std::uintptr_t, std::uint32_t,
                  std::string *) override {
    return !fail_initialize;
  }
  bool CropScale(const NvCVImage &, NvCVImage *, float, float, float, float,
                 CUstream, std::string *) noexcept override {
    return true;
  }
  bool Resize(const NvCVImage &, NvCVImage *, CUstream,
              std::string *) noexcept override {
    return true;
  }
  bool Vignette(NvCVImage *, float, float, float, CUstream,
                std::string *) noexcept override {
    return !fail_vignette;
  }
  bool Synchronize(CUstream, std::string *) noexcept override { return true; }

  bool fail_initialize = false;
  bool fail_vignette = false;
};

struct RuntimeFixture {
  vfx::VfxApi vfx;
  NvcvApi nvcv;
  ar::ArApi ar;
  cuda::CudaDriverApi cuda;
  FakeCudaState cuda_state;

  bool Initialize() {
    std::string error;
    const std::filesystem::path sdk(STUDIOCAST_FAKE_MAXINE_SDK_PATH);
    CHECK(vfx.InitializeFromLibraryPath(sdk, &error));
    CHECK(nvcv.InitializeFromLibraryPath(NvcvApi::Requirement::VfxCompat, sdk,
                                         &error));
    CHECK(ar.InitializeFromLibraryPath(sdk, &error));
    g_cuda = &cuda_state;
    CHECK(cuda.InitializeForTesting(CudaFunctions(), &error));
    CHECK(cuda.EnsureContextForDevice(0, &error));
    return true;
  }

  ProductionResidentRuntime Runtime() {
    return ProductionResidentRuntime{&vfx, &nvcv, &ar, &cuda};
  }
};

struct SetupSourceLifetime {
  RuntimeFixture runtime;
  FakeDeviceOps device_ops;
};

std::uint32_t Mask(std::initializer_list<ResidentStageKind> stages) {
  std::uint32_t mask = 0;
  for (const auto stage : stages)
    mask |= ProductionResidentStageBit(stage);
  return mask;
}

std::uint64_t Counter(FakeMaxineCounter counter) {
  return StudioCastFakeMaxineCounter(static_cast<std::size_t>(counter));
}

ProductionResidentSetup Setup(std::uint32_t mask,
                              std::uint64_t generation = 7) {
  ProductionResidentSetup setup{};
  setup.configuration_generation = generation;
  setup.width = 8;
  setup.height = 6;
  setup.output_width = 8;
  setup.output_height = 6;
  setup.enabled_maxine_stage_mask = mask;
  setup.vfx_model_directory = "/fake/models";
  setup.vignette_center_x_px = 4.0f;
  setup.vignette_center_y_px = 3.0f;
  setup.effects.engine = video::effects::EffectsEnginePreference::maxine;
  setup.effects.virtual_background.greenscreen_temporal = false;
  setup.effects.virtual_background.mode =
      (mask & ProductionResidentStageBit(ResidentStageKind::background_blur)) !=
              0
          ? video::effects::VirtualBackgroundMode::blur
          : video::effects::VirtualBackgroundMode::none;
  setup.effects.virtual_background.strength = 50;
  setup.effects.virtual_key_light.enabled =
      (mask & ProductionResidentStageBit(ResidentStageKind::relighting)) != 0;
  setup.effects.virtual_key_light.intensity = 100;
  setup.effects.virtual_key_light.hdri_path = "/fake/light.hdr";
  setup.effects.video_noise_removal.enabled = true;
  setup.effects.eye_contact.enabled = true;
  setup.effects.auto_frame.enabled = true;
  setup.effects.vignette.enabled = true;
  return setup;
}

hal::ResourceBudget Capacity() {
  return hal::ResourceBudget{1u << 20, 1u << 20, 1u << 20, 1u << 20, 64, 1};
}

hal::ImageDescriptor Image(std::uint32_t width, std::uint32_t height,
                           hal::PixelFormat format,
                           hal::ChannelOrder channel_order,
                           hal::HostCoherence coherence,
                           hal::Mutability mutability) {
  hal::ImageDescriptor image{};
  image.width = width;
  image.height = height;
  image.plane_count = 1;
  image.format = format;
  image.channel_order = channel_order;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = coherence;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  const std::uint64_t row = static_cast<std::uint64_t>(width) * 3u;
  image.planes[0] =
      hal::PlaneDescriptor{0, row, row * height, width, height, 1};
  return image;
}

video::ProductionVideoResidentSetup
SetupSourceRequest(hal::SemanticEffectId semantic,
                   const std::filesystem::path &replacement_path = {}) {
  video::ProductionVideoResidentSetup request;
  request.effects.engine = video::effects::EffectsEnginePreference::maxine;
  auto &plan = request.plan;
  plan.plan_id = 7000 + static_cast<std::uint32_t>(semantic);
  plan.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  plan.intent.backend.backend = hal::BackendId::maxine;
  plan.intent.backend.strict = true;
  plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::maxine;
  plan.intent.preferences.strict_effects_engine = true;
  auto &stage = plan.intent.stages[0];
  stage.intent.stage_id = 1;
  stage.intent.semantic = semantic;
  stage.intent.failure_action =
      hal::FailureAction::fail_open_preserve_last_valid;
  plan.intent.stage_count = 1;

  const auto signed_parameter = [&](video::CanonicalVideoParameterId id,
                                    int value) {
    stage.parameters[stage.parameter_count++] = {
        id, video::CanonicalVideoParameterKind::signed_integer, value};
  };
  const auto unsigned_parameter = [&](video::CanonicalVideoParameterId id,
                                      std::uint64_t value) {
    video::CanonicalVideoParameter parameter;
    parameter.id = id;
    parameter.kind = video::CanonicalVideoParameterKind::unsigned_integer;
    parameter.unsigned_value = value;
    stage.parameters[stage.parameter_count++] = parameter;
  };
  const auto bool_parameter = [&](video::CanonicalVideoParameterId id,
                                  bool value) {
    video::CanonicalVideoParameter parameter;
    parameter.id = id;
    parameter.kind = video::CanonicalVideoParameterKind::boolean;
    parameter.boolean_value = value;
    stage.parameters[stage.parameter_count++] = parameter;
  };

  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
    request.effects.video_noise_removal.enabled = true;
    signed_parameter(video::CanonicalVideoParameterId::strength,
                     request.effects.video_noise_removal.strength);
    break;
  case hal::SemanticEffectId::eye_contact:
    request.effects.eye_contact.enabled = true;
    signed_parameter(video::CanonicalVideoParameterId::strength,
                     request.effects.eye_contact.strength);
    bool_parameter(video::CanonicalVideoParameterId::look_away_enabled,
                   request.effects.eye_contact.look_away_enabled);
    break;
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
    request.effects.virtual_background.mode =
        semantic == hal::SemanticEffectId::virtual_background_blur
            ? video::effects::VirtualBackgroundMode::blur
            : (semantic == hal::SemanticEffectId::virtual_background_remove
                   ? video::effects::VirtualBackgroundMode::remove
                   : video::effects::VirtualBackgroundMode::replace);
    request.effects.virtual_background.greenscreen_temporal = false;
    request.effects.virtual_background.replace_path = replacement_path.string();
    signed_parameter(video::CanonicalVideoParameterId::strength,
                     request.effects.virtual_background.strength);
    unsigned_parameter(video::CanonicalVideoParameterId::remove_color_rgb, 0);
    unsigned_parameter(video::CanonicalVideoParameterId::greenscreen_mode,
                       request.effects.virtual_background.greenscreen_mode);
    bool_parameter(video::CanonicalVideoParameterId::greenscreen_temporal,
                   request.effects.virtual_background.greenscreen_temporal);
    break;
  case hal::SemanticEffectId::virtual_key_light:
    request.effects.virtual_key_light.enabled = true;
    signed_parameter(video::CanonicalVideoParameterId::intensity,
                     request.effects.virtual_key_light.intensity);
    signed_parameter(video::CanonicalVideoParameterId::temperature_preset,
                     request.effects.virtual_key_light.temperature_preset);
    signed_parameter(video::CanonicalVideoParameterId::direction_pan_degrees,
                     request.effects.virtual_key_light.direction_pan_degrees);
    break;
  case hal::SemanticEffectId::auto_frame: {
    request.effects.auto_frame.enabled = true;
    signed_parameter(video::CanonicalVideoParameterId::strength,
                     request.effects.auto_frame.strength);
    signed_parameter(video::CanonicalVideoParameterId::smoothing,
                     request.effects.auto_frame.smoothing);
    video::CanonicalVideoParameter headroom;
    headroom.id = video::CanonicalVideoParameterId::headroom;
    headroom.kind = video::CanonicalVideoParameterKind::real;
    headroom.real_value = request.effects.auto_frame.headroom;
    stage.parameters[stage.parameter_count++] = headroom;
    break;
  }
  case hal::SemanticEffectId::vignette:
    request.effects.vignette.enabled = true;
    request.effects.vignette.center_on_tracked_face = false;
    signed_parameter(video::CanonicalVideoParameterId::intensity,
                     request.effects.vignette.intensity);
    bool_parameter(video::CanonicalVideoParameterId::center_on_tracked_face,
                   request.effects.vignette.center_on_tracked_face);
    break;
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    break;
  }

  plan.host_input =
      Image(8, 6, hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  plan.host_output =
      Image(8, 6, hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  plan.invalidation_key.generations.configuration = 7;
  plan.invalidation_key.generations.geometry = 8;
  plan.invalidation_key.canonical_plan_hash =
      9000 + static_cast<std::uint32_t>(semantic);
  plan.budget = Capacity();
  plan.cpu_tails[0].id = hal::CpuTailId::backend_resolved_output;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].after_stage_id = 1;
  plan.cpu_tails[0].boundary_index = 1;
  plan.cpu_tail_count = 1;
  if (semantic == hal::SemanticEffectId::video_noise_removal ||
      semantic == hal::SemanticEffectId::eye_contact ||
      semantic == hal::SemanticEffectId::auto_frame) {
    plan.temporal.temporal = true;
    plan.temporal.requires_contiguous_capture_sequence = true;
    plan.temporal.frame_drop_behavior =
        hal::FrameDropBehavior::reset_temporal_state;
    plan.temporal.discontinuity_action =
        hal::TemporalDiscontinuityAction::reset_and_continue;
    plan.temporal.history_frames = 1;
  }
  return request;
}

hal::PreparedStage Stage(const hal::ExecutionDomainId &domain, std::uint32_t id,
                         hal::SemanticEffectId semantic,
                         hal::SemanticEffectProviderId provider,
                         bool required) {
  hal::PreparedStage stage{};
  stage.stage_id = id;
  stage.semantic = semantic;
  stage.provider = provider;
  stage.backend = hal::BackendId::maxine;
  stage.execution_domain = domain;
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image =
      Image(8, 6, hal::PixelFormat::bgr_u8, hal::ChannelOrder::bgr,
            hal::HostCoherence::not_host_visible, hal::Mutability::read_only);
  stage.output = stage.input;
  stage.output.image.mutability = hal::Mutability::read_write;
  stage.required = required;
  stage.failure_action =
      required ? hal::FailureAction::fail_closed
               : hal::FailureAction::fail_open_preserve_last_valid;
  stage.circuit_breaker.consecutive_failure_limit = 1;
  stage.circuit_breaker.action =
      required ? hal::CircuitBreakerAction::quarantine_program
               : hal::CircuitBreakerAction::disable_stage;
  stage.model.model_id_hash[0] = 0x31;
  stage.model.artifact_set_hash[0] = 0x57;
  stage.model.artifact_generation = 1;
  return stage;
}

hal::PreparedProgram CombinedProgram(const MaxineProductionHalOwner &owner) {
  hal::PreparedProgram program{};
  program.program_id = 71;
  program.owner_adapter_instance_id =
      owner.execution_domain().adapter_instance_id;
  program.execution_domain = owner.execution_domain();
  program.host_input =
      Image(8, 6, hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  program.host_output =
      Image(8, 6, hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  program.requested_backend = hal::BackendId::maxine;
  program.strict_backend_selection = true;
  program.invalidation_key.generations = hal::ProgramGenerations{
      owner.resident_key().configuration_generation,      1, 1, 1, 1,
      owner.execution_domain().logical_context_generation};
  program.invalidation_key.canonical_plan_hash = 0xAA;
  program.invalidation_key.model_artifact_hash = 0xBB;
  program.invalidation_key.execution_domain_hash = 0xCC;
  program.budget = Capacity();
  program.stages[0] = Stage(program.execution_domain, 1,
                            hal::SemanticEffectId::virtual_background_blur,
                            hal::SemanticEffectProviderId::maxine_vfx, true);
  program.stages[1] = Stage(program.execution_domain, 2,
                            hal::SemanticEffectId::virtual_key_light,
                            hal::SemanticEffectProviderId::maxine_vfx, true);
  program.stage_count = 2;
  program.cpu_tails[0].id = hal::CpuTailId::output_format_conversion;
  program.cpu_tails[0].placement =
      hal::DeclaredCpuTail::Placement::after_stage;
  program.cpu_tails[0].after_stage_id = 2;
  program.cpu_tails[0].boundary_index = 2;
  program.cpu_tail_count = 1;
  return program;
}

struct HostFrames {
  std::array<std::uint8_t, 8 * 6 * 3> input{};
  std::array<std::uint8_t, 8 * 6 * 3> output{};

  hal::HostFrameBinding Binding(const hal::PreparedProgram &program,
                                std::uint64_t sequence) {
    hal::HostFrameBinding binding{};
    binding.input_descriptor = program.host_input;
    binding.output_descriptor = program.host_output;
    binding.input_planes[0].read_data =
        reinterpret_cast<const std::byte *>(input.data());
    binding.input_planes[0].span_bytes = input.size();
    binding.input_planes[0].row_stride_bytes = 8 * 3;
    binding.output_planes[0].write_data =
        reinterpret_cast<std::byte *>(output.data());
    binding.output_planes[0].span_bytes = output.size();
    binding.output_planes[0].row_stride_bytes = 8 * 3;
    binding.input_plane_count = 1;
    binding.output_plane_count = 1;
    binding.capture_sequence = sequence;
    binding.ownership = hal::OwnershipModel::borrowed_until_completion;
    binding.lifetime = hal::ResourceLifetime::frame_borrowed;
    return binding;
  }
};

std::unique_ptr<MaxineProductionHalOwner>
CreateOwner(RuntimeFixture &runtime, FakeDeviceOps &ops,
            const ProductionResidentSetup &setup,
            MaxineProductionPreparationResult *result,
            hal::ResourceBudget capacity = Capacity()) {
  MaxineProductionBackendRequest request{};
  request.adapter_instance_id = 41;
  request.runtime = runtime.Runtime();
  request.setup = setup;
  request.capacity = capacity;
  request.device_ops = &ops;
  return MaxineProductionHalOwner::CreateBackend(request, result);
}

video::PreparedMaxineHardwareRequest BootstrapRequest(RuntimeFixture &runtime,
                                                      FakeDeviceOps &ops) {
  video::PreparedMaxineHardwareRequest request;
  request.requested_backend = hal::BackendId::maxine;
  request.strict_backend_selection = true;
  request.single_resident_island = true;
  request.host_input_is_tight_rgb24 = true;
  request.host_output_is_tight_rgb24 = true;
  request.declares_output_format_cpu_tail = true;
  // Mirrors the production camera request; the backend bootstrap owns stable
  // adapter identity assignment.
  request.adapter_instance_id = 0;
  request.width = 8;
  request.height = 6;
  request.effects = Setup(Mask({ResidentStageKind::background_blur})).effects;
  request.stages[0].stage_id = 1;
  request.stages[0].semantic = hal::SemanticEffectId::virtual_background_blur;
  request.stages[0].model.model_id_hash[0] = 0x31;
  request.stages[0].model.artifact_set_hash[0] = 0x57;
  request.stages[0].model.artifact_generation = 1;
  request.stage_count = 1;
  request.generations = {
      7, 1, 1, 1, 1, runtime.cuda.context_identity()->logical_generation};
  request.capacity = Capacity();
  request.runtime.vfx = reinterpret_cast<std::uintptr_t>(&runtime.vfx);
  request.runtime.nvcv = reinterpret_cast<std::uintptr_t>(&runtime.nvcv);
  request.runtime.ar = reinterpret_cast<std::uintptr_t>(&runtime.ar);
  request.runtime.cuda = reinterpret_cast<std::uintptr_t>(&runtime.cuda);
  request.runtime.device_ops = reinterpret_cast<std::uintptr_t>(&ops);
  request.vfx_model_directory = "/fake/models";
  return request;
}

video::PreparedMaxineHardwareRequest
MixedBootstrapRequest(RuntimeFixture &runtime, FakeDeviceOps &ops) {
  auto request = BootstrapRequest(runtime, ops);
  request.stages[1].stage_id = 2;
  request.stages[1].semantic = hal::SemanticEffectId::vignette;
  request.stage_count = 2;
  request.effects.vignette.enabled = true;
  request.effects.vignette.intensity = 40;
  request.vignette_center_x_px = 4.0F;
  request.vignette_center_y_px = 3.0F;
  return request;
}

video::VideoExecutionPlanRequest
CoordinatorRequest(const hal::ExecutionDomainId &domain,
                   std::uint64_t plan_id) {
  video::VideoExecutionPlanRequest request;
  request.plan_id = plan_id;
  request.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  request.intent.backend.backend = hal::BackendId::maxine;
  request.intent.backend.strict = true;
  request.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  request.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::maxine;
  request.intent.preferences.strict_effects_engine = true;
  request.intent.requested_provider = hal::SemanticEffectProviderId::maxine_vfx;
  request.intent.strict_provider_selection = true;
  request.intent.stages[0].intent.stage_id = 1;
  request.intent.stages[0].intent.semantic =
      hal::SemanticEffectId::virtual_background_blur;
  request.intent.stages[0].intent.failure_action =
      hal::FailureAction::fail_open_preserve_last_valid;
  request.intent.stages[0].parameters[0] = {
      video::CanonicalVideoParameterId::strength,
      video::CanonicalVideoParameterKind::signed_integer,
      50,
  };
  request.intent.stages[0].parameter_count = 1;
  request.intent.stages[0].model.model_id_hash[0] = 0x31;
  request.intent.stages[0].model.artifact_set_hash[0] = 0x57;
  request.intent.stages[0].model.artifact_generation = 1;
  request.intent.stage_count = 1;
  request.host_input =
      Image(8, 6, hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  request.host_output =
      Image(8, 6, hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  request.invalidation_key.generations = {
      7, 1, 1, 1, 1, domain.logical_context_generation};
  request.invalidation_key.canonical_plan_hash = plan_id + 100;
  request.invalidation_key.model_artifact_hash = 0xBB;
  request.invalidation_key.execution_domain_hash = 0xCC;
  request.budget = Capacity();
  request.cpu_tails[0].id = hal::CpuTailId::output_format_conversion;
  request.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  request.cpu_tails[0].after_stage_id = 1;
  request.cpu_tails[0].boundary_index = 1;
  request.cpu_tail_count = 1;
  return request;
}

video::VideoExecutionPlanRequest
MixedCoordinatorRequest(const hal::ExecutionDomainId &domain,
                        std::uint64_t plan_id) {
  auto request = CoordinatorRequest(domain, plan_id);
  request.intent.requested_provider = hal::SemanticEffectProviderId::invalid;
  request.intent.strict_provider_selection = false;
  request.intent.stages[1].intent.stage_id = 2;
  request.intent.stages[1].intent.semantic = hal::SemanticEffectId::vignette;
  request.intent.stages[1].intent.failure_action =
      hal::FailureAction::fail_open_preserve_last_valid;
  request.intent.stages[1].parameters[0] = {
      video::CanonicalVideoParameterId::intensity,
      video::CanonicalVideoParameterKind::signed_integer,
      40,
  };
  request.intent.stages[1].parameter_count = 1;
  request.intent.stage_count = 2;
  request.cpu_tails[0].after_stage_id = 2;
  request.cpu_tails[0].boundary_index = 2;
  return request;
}

bool ExactDomainAndPreparedExecutionAreAllocationFree() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionPreparationResult preparation{};
  auto owner = CreateOwner(runtime, ops,
                           Setup(Mask({ResidentStageKind::background_blur,
                                       ResidentStageKind::relighting})),
                           &preparation);
  if (!owner)
    std::cerr << "owner preparation failed: status="
              << static_cast<unsigned>(preparation.status)
              << " reason=" << static_cast<unsigned>(preparation.reason)
              << " diagnostic=" << preparation.diagnostic << '\n';
  CHECK(owner);
  CHECK(preparation.status == hal::StatusId::ready);
  CHECK(hal::ValidateExecutionDomain(owner->execution_domain()));
  CHECK(owner->execution_domain().backend == hal::BackendId::maxine);
  CHECK(owner->execution_domain().api_family == hal::ApiFamily::maxine_nvcv);
  CHECK(owner->execution_domain().physical_device.stable_id[0] == 0x42);
  CHECK(owner->execution_domain().logical_context_generation ==
        runtime.cuda.context_identity()->logical_generation);
  CHECK(owner->execution_domain().queue_or_stream_identity ==
        owner->executor()->stream_identity());
  CHECK(owner->execution_domain().queue_or_stream_identity == 0x30);
  CHECK(owner->execution_domain().adapter_instance_id == 41);

  auto program = CombinedProgram(*owner);
  CHECK(owner->ConfigureProgram(program, &preparation) == hal::StatusId::ready);
  HostFrames frames{};
  frames.input.fill(0x5A);

  // Warm every path before allocation accounting.
  hal::HostFrameTokens tokens{};
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(owner->session()->BindHostFrame(frames.Binding(program, 1), &tokens,
                                        &reason) == hal::StatusId::ready);
  hal::ExecuteRequest request{};
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = 1;
  request.input = tokens.input;
  auto warm = owner->session()->Execute(request);
  CHECK(warm.status == hal::StatusId::executed);
  hal::HostFrameOutputResolution resolution{};
  CHECK(owner->session()->ResolveHostFrameOutput(warm.completion, warm.output,
                                                 &resolution, &reason) ==
        hal::StatusId::ready);
  CHECK(owner->session()->CommitOutput({warm.completion, true}, &reason) ==
        hal::StatusId::ready);

  owner->executor()->ResetTelemetry();
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool ok = true;
  for (std::uint64_t sequence = 2; sequence != 130; ++sequence) {
    if (owner->session()->BindHostFrame(frames.Binding(program, sequence),
                                        &tokens,
                                        &reason) != hal::StatusId::ready) {
      ok = false;
      break;
    }
    request.capture_sequence = sequence;
    request.input = tokens.input;
    const auto result = owner->session()->Execute(request);
    if (result.status != hal::StatusId::executed ||
        result.transfers.uploads != 1 ||
        result.transfers.final_readbacks != 1 ||
        result.transfers.intermediate_uploads != 0 ||
        result.transfers.intermediate_readbacks != 0 ||
        owner->session()->ResolveHostFrameOutput(
            result.completion, result.output, &resolution, &reason) !=
            hal::StatusId::ready ||
        owner->session()->CommitOutput({result.completion, true}, &reason) !=
            hal::StatusId::ready) {
      ok = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(ok);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  const auto &telemetry = owner->executor()->telemetry();
  CHECK(telemetry
            .transfers[static_cast<std::size_t>(
                ProductionTransferKind::host_upload)]
            .successes == 128);
  CHECK(telemetry
            .transfers[static_cast<std::size_t>(
                ProductionTransferKind::cpu_continuation_download)]
            .successes == 128);
  CHECK(telemetry.matte_inferences.successes == 128);
  CHECK(telemetry.shared_matte_reuses == 128);
  const auto conversion = owner->output_format_conversion_telemetry();
  CHECK(conversion.attempts == 129);
  CHECK(conversion.successes == 129);
  CHECK(owner->session()->latest_committed_frame().valid);
  CHECK(owner->session()->latest_committed_frame().stage_count == 2);
  CHECK(owner->session()->latest_committed_frame().stages[0].backend ==
        hal::BackendId::maxine);

  const auto retained_program_id = owner->session()->program().program_id;
  auto reversed = program;
  ++reversed.program_id;
  std::swap(reversed.stages[0], reversed.stages[1]);
  CHECK(owner->ConfigureProgram(reversed, &preparation) ==
        hal::StatusId::blocked);
  CHECK(preparation.reason == hal::ReasonId::frame_order_violated);
  CHECK(owner->session()->program().program_id == retained_program_id);

  auto mutually_exclusive = program;
  ++mutually_exclusive.program_id;
  mutually_exclusive.stages[2] = mutually_exclusive.stages[1];
  mutually_exclusive.stages[2].stage_id = 3;
  mutually_exclusive.stages[1] =
      Stage(program.execution_domain, 2,
            hal::SemanticEffectId::virtual_background_remove,
            hal::SemanticEffectProviderId::maxine_vfx, true);
  mutually_exclusive.stage_count = 3;
  CHECK(owner->ConfigureProgram(mutually_exclusive, &preparation) ==
        hal::StatusId::blocked);
  CHECK(preparation.reason == hal::ReasonId::malformed_contract);
  CHECK(owner->session()->program().program_id == retained_program_id);
  return true;
}

bool IdentityStageBudgetAndRollbackFailuresAreTyped() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionPreparationResult result{};
  const auto setup = Setup(Mask({ResidentStageKind::vignette}));
  hal::ResourceBudget minimum{};
  CHECK(KnownMinimumMaxineBudget(setup, &minimum));
  auto exhausted = minimum;
  --exhausted.persistent_bytes;
  CHECK(!CreateOwner(runtime, ops, setup, &result, exhausted));
  CHECK(result.status == hal::StatusId::resource_exhausted);
  CHECK(result.reason == hal::ReasonId::resource_exhausted);

  auto wrong_identity = setup;
  wrong_identity.runtime_identity = 999;
  CHECK(!CreateOwner(runtime, ops, wrong_identity, &result));
  CHECK(result.reason == hal::ReasonId::wrong_generation);

  auto owner = CreateOwner(runtime, ops, setup, &result);
  CHECK(owner);
  auto program = CombinedProgram(*owner);
  program.stage_count = 1;
  program.cpu_tails[0].after_stage_id = 1;
  program.cpu_tails[0].boundary_index = 1;
  program.stages[0] =
      Stage(program.execution_domain, 1, hal::SemanticEffectId::vignette,
            hal::SemanticEffectProviderId::maxine_cuda_utility, true);
  CHECK(owner->ConfigureProgram(program, &result) == hal::StatusId::ready);
  const auto retained_program_id = owner->session()->program().program_id;

  auto wrong_stream = program;
  ++wrong_stream.program_id;
  ++wrong_stream.execution_domain.queue_or_stream_identity;
  wrong_stream.stages[0].execution_domain = wrong_stream.execution_domain;
  CHECK(owner->ConfigureProgram(wrong_stream, &result) ==
        hal::StatusId::blocked);
  CHECK(result.reason == hal::ReasonId::wrong_execution_domain);
  CHECK(owner->session()->program().program_id == retained_program_id);

  auto wrong_generation = program;
  ++wrong_generation.program_id;
  ++wrong_generation.invalidation_key.generations.configuration;
  CHECK(owner->ConfigureProgram(wrong_generation, &result) ==
        hal::StatusId::blocked);
  CHECK(result.reason == hal::ReasonId::wrong_generation);
  CHECK(owner->session()->program().program_id == retained_program_id);

  auto unsupported = program;
  ++unsupported.program_id;
  unsupported.stages[0] =
      Stage(program.execution_domain, 1, hal::SemanticEffectId::mirror,
            hal::SemanticEffectProviderId::maxine_cuda_utility, true);
  CHECK(owner->ConfigureProgram(unsupported, &result) ==
        hal::StatusId::blocked);
  CHECK(result.reason == hal::ReasonId::semantic_effect_unsupported);
  CHECK(owner->session()->program().program_id == retained_program_id);

  auto wrong_provider = program;
  ++wrong_provider.program_id;
  wrong_provider.stages[0].provider = hal::SemanticEffectProviderId::maxine_vfx;
  CHECK(owner->ConfigureProgram(wrong_provider, &result) ==
        hal::StatusId::blocked);
  CHECK(result.reason == hal::ReasonId::semantic_provider_unavailable);
  CHECK(owner->session()->program().program_id == retained_program_id);

  return true;
}

bool OptionalFailureCommitDeviceLossAndShutdownAreSafe() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionPreparationResult result{};
  auto owner = CreateOwner(runtime, ops,
                           Setup(Mask({ResidentStageKind::vignette})), &result);
  CHECK(owner);
  auto program = CombinedProgram(*owner);
  program.stage_count = 1;
  program.cpu_tails[0].after_stage_id = 1;
  program.cpu_tails[0].boundary_index = 1;
  program.stages[0] =
      Stage(program.execution_domain, 1, hal::SemanticEffectId::vignette,
            hal::SemanticEffectProviderId::maxine_cuda_utility, false);
  CHECK(owner->ConfigureProgram(program, &result) == hal::StatusId::ready);

  HostFrames frames{};
  frames.input.fill(0x33);
  hal::HostFrameTokens tokens{};
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(owner->session()->BindHostFrame(frames.Binding(program, 1), &tokens,
                                        &reason) == hal::StatusId::ready);
  hal::ExecuteRequest request{};
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = 1;
  request.input = tokens.input;
  ops.fail_vignette = true;
  const auto degraded = owner->session()->Execute(request);
  if (degraded.status != hal::StatusId::degraded)
    std::cerr << "optional execution failed: status="
              << static_cast<unsigned>(degraded.status)
              << " reason=" << static_cast<unsigned>(degraded.reason)
              << " stage_status="
              << static_cast<unsigned>(degraded.stages[0].status)
              << " stage_reason="
              << static_cast<unsigned>(degraded.stages[0].reason) << '\n';
  CHECK(degraded.status == hal::StatusId::degraded);
  CHECK(degraded.reason == hal::ReasonId::optional_stage_failed_open);
  CHECK(degraded.output_valid);
  CHECK(degraded.stages[0].valid_output_preserved);
  CHECK(!owner->session()->latest_committed_frame().valid);
  hal::HostFrameOutputResolution resolution{};
  CHECK(owner->session()->ResolveHostFrameOutput(
            degraded.completion, degraded.output, &resolution, &reason) ==
        hal::StatusId::ready);
  CHECK(owner->session()->CommitOutput({degraded.completion, false}, &reason) ==
        hal::StatusId::failed);
  CHECK(reason == hal::ReasonId::output_write_failed);
  CHECK(!owner->session()->latest_committed_frame().valid);

  owner->NotifyDeviceLost();
  CHECK(owner->device_lost());
  CHECK(owner->session()->quarantined());
  CHECK(owner->session()->Execute(request).status ==
        hal::StatusId::quarantined);
  owner->Shutdown();
  CHECK(owner->stopped());
  CHECK(owner->session()->stopped());
  CHECK(owner->session()->Execute(request).status == hal::StatusId::stopped);
  CHECK(owner->adapter() == nullptr);
  CHECK(owner->executor() == nullptr);
  return true;
}

bool ProductionBootstrapReusesOwnerRollsBackAndRetainsExecutorLifetime() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionVideoExecutionBootstrapRequest bootstrap_request;
  bootstrap_request.prepared_request = BootstrapRequest(runtime, ops);
  bootstrap_request.factory_id = 9101;
  bootstrap_request.selection_priority = 10;
  bootstrap_request.minimum_synchronizations_per_frame = 1;
  bootstrap_request.maximum_synchronizations_per_frame = 16;
  MaxineProductionVideoExecutionBootstrapResult bootstrap_result;
  auto bootstrap = MaxineProductionVideoExecutionBootstrap::Create(
      bootstrap_request, &bootstrap_result);
  CHECK(bootstrap);
  CHECK(bootstrap_result.status == hal::StatusId::ready);
  CHECK(bootstrap->adapter_registration().adapter_instance_id != 0);
  CHECK(bootstrap->adapter_registration().resident_adapter != nullptr);
  CHECK(bootstrap->adapter_registration().resident_adapter !=
        bootstrap->production_owner()->adapter());
  CHECK(bootstrap->adapter_registration()
            .resident_adapter->adapter_instance_id() ==
        bootstrap->production_owner()->adapter()->adapter_instance_id());
  CHECK(bootstrap->adapter_registration().execution_domain ==
        bootstrap->production_owner()->execution_domain());
  CHECK(bootstrap->adapter_registration().semantic_capability_count == 1);
  CHECK(bootstrap->adapter_registration().semantic_capabilities[0].cpu_tail ==
        hal::CpuTailId::output_format_conversion);
  auto *const native_executor = bootstrap->production_owner()->executor();
  CHECK(native_executor != nullptr);

  hal::StaticAdapterRegistry registry;
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(registry.Register(bootstrap->adapter_registration(), &reason) ==
        hal::StatusId::ready);
  video::VideoExecutionCoordinator coordinator(registry);
  CHECK(coordinator.RegisterFactory(bootstrap->factory_registration(),
                                    &reason) == hal::StatusId::ready);
  const auto domain = bootstrap->adapter_registration().execution_domain;
  auto request = CoordinatorRequest(domain, 1001);
  video::VideoExecutionPrepareEvidence prepare;
  CHECK(coordinator.Prepare(request, &prepare, &reason) ==
        hal::StatusId::degraded);
  CHECK(prepare.selected_factory_id == bootstrap_request.factory_id);
  CHECK(bootstrap->production_owner()->executor() == native_executor);
  CHECK(!bootstrap->production_owner()->stopped());

  auto replacement = CoordinatorRequest(domain, 1002);
  CHECK(coordinator.Prepare(replacement, &prepare, &reason) ==
        hal::StatusId::degraded);
  CHECK(bootstrap->production_owner()->executor() == native_executor);
  CHECK(!bootstrap->production_owner()->stopped());

  auto malformed_candidate = coordinator.Snapshot().plan;
  ++malformed_candidate.prepared_program.execution_domain
        .logical_context_generation;
  video::VideoAdapterPrepareEvidence failed_prepare;
  CHECK(!bootstrap->factory().Prepare(malformed_candidate, &failed_prepare));
  CHECK(failed_prepare.status == hal::StatusId::blocked);
  CHECK(failed_prepare.reason == hal::ReasonId::malformed_contract);
  CHECK(failed_prepare.candidate_resources_rolled_back);
  CHECK(coordinator.Snapshot().plan.plan_id == replacement.plan_id);
  CHECK(bootstrap->production_owner()->executor() == native_executor);

  HostFrames frames{};
  frames.input.fill(0x62);
  auto binding =
      frames.Binding(CombinedProgram(*bootstrap->production_owner()), 1);
  binding.input_descriptor = replacement.host_input;
  binding.output_descriptor = replacement.host_output;
  const auto frame = coordinator.Execute(binding);
  CHECK(frame.status == hal::StatusId::executed);
  CHECK(frame.output_ready);
  CHECK(frame.transfers.uploads == 1);
  CHECK(frame.transfers.final_readbacks == 1);
  CHECK(frame.transfers.intermediate_uploads == 0);
  CHECK(frame.transfers.intermediate_readbacks == 0);
  CHECK(frame.transfers.cpu_tail_boundaries == 1);
  CHECK(coordinator.CommitOutput(frame.completion, true, &reason) ==
        hal::StatusId::ready);
  CHECK(coordinator.Snapshot().latest_committed_frame.valid);
  CHECK(coordinator.Snapshot().telemetry.active_backend ==
        hal::BackendId::maxine);

  // The prepared executor retains the production state. The coordinator must
  // be able to tear its per-plan slot down after the external bootstrap and
  // factory references have gone away.
  bootstrap.reset();
  coordinator.Shutdown();
  CHECK(coordinator.Snapshot().stopped);
  return true;
}

bool TypedSetupLeasePublishesExactMixedEvidenceAndTearsDownInOrder() {
  StudioCastFakeMaxineResetCounters();
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionVideoExecutionBootstrapRequest request;
  request.prepared_request = MixedBootstrapRequest(runtime, ops);
  request.factory_id = 9201;
  request.selection_priority = 11;
  request.minimum_synchronizations_per_frame = 1;
  request.maximum_synchronizations_per_frame = 16;

  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence acquire;
  MaxineProductionVideoExecutionBootstrapResult result;
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(MaxineProductionVideoExecutionBootstrap::AcquireSetupLease(
            request, &lease, &acquire, &result, &reason) ==
        hal::StatusId::ready);
  CHECK(reason == hal::ReasonId::none);
  CHECK(lease.live());
  CHECK(acquire.status == hal::StatusId::ready);
  CHECK(!acquire.candidate_owner_rolled_back);
  CHECK(result.status == hal::StatusId::ready);

  const auto &snapshot = lease.snapshot();
  CHECK(snapshot.contract_version ==
        video::kProductionVideoSetupLeaseContractVersion);
  CHECK(snapshot.state == video::ProductionVideoSetupLeaseState::live);
  CHECK(snapshot.backend == hal::BackendId::maxine);
  CHECK(snapshot.execution_domain == result.execution_domain);
  CHECK(snapshot.generations == request.prepared_request.generations);
  CHECK(snapshot.generations.context ==
        snapshot.execution_domain.logical_context_generation);
  CHECK(snapshot.reserved_budget.persistent_bytes ==
        request.prepared_request.capacity.persistent_bytes);
  CHECK(snapshot.reserved_budget.scratch_bytes ==
        request.prepared_request.capacity.scratch_bytes);
  CHECK(snapshot.reserved_budget.upload_staging_bytes ==
        request.prepared_request.capacity.upload_staging_bytes);
  CHECK(snapshot.reserved_budget.readback_staging_bytes ==
        request.prepared_request.capacity.readback_staging_bytes);
  CHECK(snapshot.reserved_budget.resource_count ==
        request.prepared_request.capacity.resource_count);
  CHECK(snapshot.reserved_budget.in_flight_slots == 1);
  CHECK(snapshot.provider_count == 2);
  CHECK(snapshot.providers[0] == hal::SemanticEffectProviderId::maxine_vfx);
  CHECK(snapshot.providers[1] ==
        hal::SemanticEffectProviderId::maxine_cuda_utility);
  CHECK(snapshot.factory.provider == hal::SemanticEffectProviderId::invalid);
  CHECK(snapshot.readiness.provider == hal::SemanticEffectProviderId::invalid);
  CHECK(snapshot.readiness.ready_stage_mask == 0x3);
  CHECK(snapshot.readiness.available);
  CHECK(snapshot.readiness.evidence_count == 5);
  CHECK(snapshot.artifact_count == 1);
  CHECK(snapshot.artifacts[0].semantic ==
        hal::SemanticEffectId::virtual_background_blur);
  CHECK(snapshot.artifacts[0].provider ==
        hal::SemanticEffectProviderId::maxine_vfx);
  CHECK(snapshot.artifacts[0].artifact.model_id_hash ==
        request.prepared_request.stages[0].model.model_id_hash);
  CHECK(snapshot.artifacts[0].artifact.artifact_set_hash ==
        request.prepared_request.stages[0].model.artifact_set_hash);
  CHECK(snapshot.artifacts[0].artifact.artifact_generation ==
        snapshot.generations.model);
  CHECK(snapshot.adapter.resident_adapter != nullptr);
  CHECK(snapshot.factory.factory != nullptr);

  auto *const stable_adapter = snapshot.adapter.resident_adapter;
  auto *const stable_factory = snapshot.factory.factory;
  const auto stable_adapter_id = stable_adapter->adapter_instance_id();
  const auto stream_destroys_before =
      g_stream_destroy_calls.load(std::memory_order_relaxed);
  const auto vfx_destroys_before = Counter(FakeMaxineCounter::vfx_destroy);

  hal::StaticAdapterRegistry registry;
  CHECK(registry.Register(snapshot.adapter, &reason) == hal::StatusId::ready);
  video::VideoExecutionCoordinator coordinator(registry);
  CHECK(coordinator.RegisterFactory(snapshot.factory, &reason) ==
        hal::StatusId::ready);
  auto coordinator_request =
      MixedCoordinatorRequest(snapshot.execution_domain, 2001);
  video::VideoExecutionPrepareEvidence prepare;
  const auto prepare_status =
      coordinator.Prepare(coordinator_request, &prepare, &reason);
  if (prepare_status != hal::StatusId::ready &&
      prepare_status != hal::StatusId::degraded) {
    const auto preflight = snapshot.factory.factory->Preflight(
        coordinator_request, prepare.resolution, prepare.domain_plan,
        video::VideoExecutionRoute::prepared_resident);
    std::cerr << "typed lease prepare failed: status="
              << static_cast<unsigned>(prepare_status)
              << " reason=" << static_cast<unsigned>(reason)
              << " evidence_reason=" << static_cast<unsigned>(prepare.reason)
              << " factory_attempted=" << prepare.prepared_factory_attempted
              << " rolled_back=" << prepare.candidate_resources_rolled_back
              << " resolution_status="
              << static_cast<unsigned>(prepare.resolution.status)
              << " domain_status="
              << static_cast<unsigned>(prepare.domain_plan.status)
              << " preflight_status=" << static_cast<unsigned>(preflight.status)
              << " preflight_reason=" << static_cast<unsigned>(preflight.reason)
              << " transfers=" << prepare.domain_plan.expected_transfers.uploads
              << ',' << prepare.domain_plan.expected_transfers.final_readbacks
              << ','
              << prepare.domain_plan.expected_transfers.cpu_tail_boundaries
              << '\n';
  }
  CHECK(prepare_status == hal::StatusId::ready ||
        prepare_status == hal::StatusId::degraded);

  // The coordinator releases its prepared slot first. Native bootstrap
  // resources remain alive until the typed lease is shut down.
  coordinator.Shutdown();
  CHECK(g_stream_destroy_calls.load(std::memory_order_relaxed) ==
        stream_destroys_before);
  lease.Shutdown();
  CHECK(!lease.live());
  CHECK(lease.snapshot().state ==
        video::ProductionVideoSetupLeaseState::stopped);
  CHECK(lease.snapshot().adapter.resident_adapter == stable_adapter);
  CHECK(lease.snapshot().factory.factory == stable_factory);
  CHECK(stable_adapter->adapter_instance_id() == stable_adapter_id);
  CHECK(stable_adapter->backend_id() == hal::BackendId::maxine);
  hal::PreparedProgramHandle stopped_candidate;
  hal::ReadinessSnapshot stopped_readiness;
  CHECK(stable_adapter->Prepare({}, &stopped_candidate, &stopped_readiness,
                                &reason) == hal::StatusId::stopped);
  CHECK(reason == hal::ReasonId::shutdown_requested);
  CHECK(g_stream_destroy_calls.load(std::memory_order_relaxed) ==
        stream_destroys_before + 1);
  CHECK(Counter(FakeMaxineCounter::vfx_destroy) > vfx_destroys_before);

  const auto vfx_destroys_after = Counter(FakeMaxineCounter::vfx_destroy);
  lease.Shutdown();
  CHECK(g_stream_destroy_calls.load(std::memory_order_relaxed) ==
        stream_destroys_before + 1);
  CHECK(Counter(FakeMaxineCounter::vfx_destroy) == vfx_destroys_after);
  return true;
}

bool TypedSetupLeaseRollsBackStaleEvidenceAndRetainsPrevious() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionVideoExecutionBootstrapRequest request;
  request.prepared_request = BootstrapRequest(runtime, ops);
  request.factory_id = 9301;
  request.selection_priority = 12;
  request.minimum_synchronizations_per_frame = 1;
  request.maximum_synchronizations_per_frame = 16;

  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence acquire;
  MaxineProductionVideoExecutionBootstrapResult result;
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(MaxineProductionVideoExecutionBootstrap::AcquireSetupLease(
            request, &lease, &acquire, &result, &reason) ==
        hal::StatusId::ready);
  auto *const retained_adapter = lease.snapshot().adapter.resident_adapter;
  auto *const retained_factory = lease.snapshot().factory.factory;

  auto stale_context = request;
  stale_context.factory_id = 9302;
  ++stale_context.prepared_request.generations.context;
  const auto stream_destroys_before =
      g_stream_destroy_calls.load(std::memory_order_relaxed);
  CHECK(MaxineProductionVideoExecutionBootstrap::AcquireSetupLease(
            stale_context, &lease, &acquire, &result, &reason) ==
        hal::StatusId::stale);
  CHECK(reason == hal::ReasonId::wrong_generation);
  CHECK(acquire.candidate_owner_rolled_back);
  CHECK(acquire.previous_lease_retained);
  CHECK(lease.live());
  CHECK(lease.snapshot().adapter.resident_adapter == retained_adapter);
  CHECK(lease.snapshot().factory.factory == retained_factory);
  CHECK(g_stream_destroy_calls.load(std::memory_order_relaxed) ==
        stream_destroys_before + 1);

  auto wrong_artifact_generation = request;
  wrong_artifact_generation.factory_id = 9303;
  ++wrong_artifact_generation.prepared_request.generations.model;
  CHECK(MaxineProductionVideoExecutionBootstrap::AcquireSetupLease(
            wrong_artifact_generation, &lease, &acquire, &result, &reason) ==
        hal::StatusId::stale);
  CHECK(reason == hal::ReasonId::wrong_generation);
  CHECK(acquire.candidate_owner_rolled_back);
  CHECK(acquire.previous_lease_retained);
  CHECK(lease.live());
  CHECK(lease.snapshot().adapter.resident_adapter == retained_adapter);
  CHECK(lease.snapshot().factory.factory == retained_factory);

  lease.Shutdown();
  return true;
}

bool BackendSetupSourceBindsEveryCurrentSemanticExactly() {
  auto lifetime = std::make_shared<SetupSourceLifetime>();
  CHECK(lifetime->runtime.Initialize());
  const auto runtime = lifetime->runtime.Runtime();
  MaxineProductionBackendSetupSourceConfiguration configuration;
  configuration.sdk_library_path = STUDIOCAST_FAKE_MAXINE_SDK_PATH;
  configuration.vfx_model_directory = "/fake/models";
  configuration.injected_runtime = {
      reinterpret_cast<std::uintptr_t>(runtime.vfx),
      reinterpret_cast<std::uintptr_t>(runtime.nvcv),
      reinterpret_cast<std::uintptr_t>(runtime.ar),
      reinterpret_cast<std::uintptr_t>(runtime.cuda),
      reinterpret_cast<std::uintptr_t>(&lifetime->device_ops),
  };
  configuration.injected_runtime_lifetime = lifetime;
  configuration.bypass_system_diagnostics_for_testing = true;
  auto source = CreateMaxineProductionVideoBackendSetupSource(configuration);
  CHECK(source);

  const auto replacement = std::filesystem::temp_directory_path() /
                           "studiocast-maxine-setup-source-replacement.ppm";
  {
    std::ofstream image(replacement, std::ios::binary | std::ios::trunc);
    CHECK(image.good());
    image << "P6\n2 1\n255\n";
    const std::array<char, 6> pixels{char(0xff), 0, 0, 0, char(0xff), 0};
    image.write(pixels.data(), static_cast<std::streamsize>(pixels.size()));
    CHECK(image.good());
  }

  const std::array semantics{
      hal::SemanticEffectId::video_noise_removal,
      hal::SemanticEffectId::eye_contact,
      hal::SemanticEffectId::virtual_background_blur,
      hal::SemanticEffectId::virtual_background_remove,
      hal::SemanticEffectId::virtual_background_replace,
      hal::SemanticEffectId::virtual_key_light,
      hal::SemanticEffectId::auto_frame,
      hal::SemanticEffectId::vignette,
  };
  for (const auto semantic : semantics) {
    const auto request = SetupSourceRequest(semantic, replacement);
    auto result = source->AcquireResidentSetup(request);
    CHECK(result.setup_completed_outside_frame_loop);
    CHECK(result.status == hal::StatusId::ready);
    CHECK(result.reason == hal::ReasonId::none);
    CHECK(result.canonical_effects == request.effects);
    CHECK(result.lease.live());
    CHECK(result.plan.intent.stage_count == 1);
    CHECK(result.plan.intent.stages[0].intent.semantic == semantic);
    CHECK(result.plan.invalidation_key.execution_domain_hash != 0);
    CHECK(result.lease.snapshot().backend == hal::BackendId::maxine);
    CHECK(result.lease.snapshot().execution_domain.backend ==
          hal::BackendId::maxine);
    CHECK(result.lease.snapshot().adapter.semantic_capability_count == 1);
    CHECK(result.lease.snapshot().adapter.semantic_capabilities[0].cpu_tail ==
          hal::CpuTailId::output_format_conversion);
    CHECK(result.plan.host_output.format == hal::PixelFormat::rgb_u8);
    CHECK(result.plan.host_output.channel_order == hal::ChannelOrder::rgb);
    CHECK(result.plan.cpu_tail_count == 1);
    CHECK(result.plan.cpu_tails[0].id ==
          hal::CpuTailId::output_format_conversion);
    CHECK(result.lease.snapshot().reserved_budget.persistent_bytes ==
          request.plan.budget.persistent_bytes);
    const bool has_configurable_model =
        semantic == hal::SemanticEffectId::video_noise_removal ||
        semantic == hal::SemanticEffectId::eye_contact ||
        semantic == hal::SemanticEffectId::virtual_background_blur ||
        semantic == hal::SemanticEffectId::virtual_background_remove ||
        semantic == hal::SemanticEffectId::virtual_background_replace ||
        semantic == hal::SemanticEffectId::auto_frame;
    CHECK(result.model_selection_count ==
          static_cast<std::size_t>(has_configurable_model));
    if (has_configurable_model) {
      CHECK(result.model_selections[0].semantic == semantic);
      CHECK(result.model_selections[0].requested_model_id.empty());
      CHECK(!result.model_selections[0].selected_model_id.empty());
      CHECK(result.model_selections[0].default_selected);
    }
    if (semantic == hal::SemanticEffectId::vignette) {
      CHECK(result.lease.snapshot().artifact_count == 0);
    } else {
      CHECK(result.plan.intent.stages[0].model.artifact_generation != 0);
      CHECK(result.lease.snapshot().artifact_count == 1);
      CHECK(result.lease.snapshot().artifacts[0].semantic == semantic);
      const auto &bound = result.lease.snapshot().artifacts[0].artifact;
      const auto &planned = result.plan.intent.stages[0].model;
      CHECK(bound.model_id_hash == planned.model_id_hash);
      CHECK(bound.artifact_set_hash == planned.artifact_set_hash);
      CHECK(bound.artifact_generation == planned.artifact_generation);
    }
    const auto shutdown = result.lease.Shutdown();
    CHECK(shutdown.status == hal::StatusId::stopped);
    CHECK(shutdown.quiescent);
    CHECK(shutdown.driver_context_release_permitted);
  }
  auto wrong_backend =
      SetupSourceRequest(hal::SemanticEffectId::virtual_background_blur);
  wrong_backend.plan.intent.backend.backend = hal::BackendId::cuda;
  const auto rejected = source->AcquireResidentSetup(wrong_backend);
  CHECK(rejected.status == hal::StatusId::blocked);
  CHECK(rejected.reason == hal::ReasonId::strict_backend_unavailable);
  auto automatic =
      SetupSourceRequest(hal::SemanticEffectId::virtual_background_blur);
  automatic.effects.engine =
      video::effects::EffectsEnginePreference::auto_select;
  automatic.plan.intent.backend.mode = hal::BackendRequestMode::automatic;
  automatic.plan.intent.backend.backend = hal::BackendId::invalid;
  automatic.plan.intent.backend.strict = false;
  automatic.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  automatic.plan.intent.preferences.strict_effects_engine = false;
  auto automatic_result = source->AcquireResidentSetup(automatic);
  CHECK(automatic_result.status == hal::StatusId::ready);
  CHECK(automatic_result.canonical_effects == automatic.effects);
  CHECK(automatic_result.lease.Shutdown().status == hal::StatusId::stopped);
  // Production requests preserve the Maxine-only default without imposing its
  // history on CUDA. The selected Maxine source must add that history exactly.
  for (const bool use_automatic : {false, true}) {
    auto temporal = use_automatic ? automatic :
        SetupSourceRequest(hal::SemanticEffectId::virtual_background_blur);
    temporal.effects.virtual_background.greenscreen_temporal = true;
    temporal.plan.intent.stages[0].parameters[3].boolean_value = true;
    CHECK(!temporal.plan.temporal.temporal);
    auto temporal_result = source->AcquireResidentSetup(temporal);
    CHECK(temporal_result.status == hal::StatusId::ready);
    CHECK(temporal_result.canonical_effects == temporal.effects);
    CHECK(temporal_result.plan.temporal.temporal);
    CHECK(temporal_result.plan.temporal.requires_contiguous_capture_sequence);
    CHECK(temporal_result.plan.temporal.history_frames == 1);
    CHECK(temporal_result.plan.temporal.frame_drop_behavior ==
          hal::FrameDropBehavior::reset_temporal_state);
    CHECK(temporal_result.lease.Shutdown().status == hal::StatusId::stopped);
  }
  std::error_code cleanup_error;
  std::filesystem::remove(replacement, cleanup_error);
  CHECK(!cleanup_error);
  return true;
}

bool TemporalConfigurationsPrepareButResetRequiresReconstruction() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  MaxineProductionPreparationResult result{};
  auto setup = Setup(Mask({ResidentStageKind::background_blur}));
  setup.effects.virtual_background.greenscreen_temporal = true;
  auto owner = CreateOwner(runtime, ops, setup, &result);
  CHECK(owner);
  auto program = CombinedProgram(*owner);
  program.stage_count = 1;
  program.cpu_tails[0].after_stage_id = 1;
  program.cpu_tails[0].boundary_index = 1;
  program.stages[0] = Stage(program.execution_domain, 1,
                            hal::SemanticEffectId::virtual_background_blur,
                            hal::SemanticEffectProviderId::maxine_vfx, true);
  program.temporal.temporal = true;
  program.temporal.requires_contiguous_capture_sequence = true;
  program.temporal.history_frames = 1;
  CHECK(owner->ConfigureProgram(program, &result) == hal::StatusId::ready);
  CHECK(result.reason == hal::ReasonId::none);
  CHECK(owner->prepared());
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
  if (!ExactDomainAndPreparedExecutionAreAllocationFree() ||
      !IdentityStageBudgetAndRollbackFailuresAreTyped() ||
      !OptionalFailureCommitDeviceLossAndShutdownAreSafe() ||
      !ProductionBootstrapReusesOwnerRollsBackAndRetainsExecutorLifetime() ||
      !TypedSetupLeasePublishesExactMixedEvidenceAndTearsDownInOrder() ||
      !TypedSetupLeaseRollsBackStaleEvidenceAndRetainsPrevious() ||
      !BackendSetupSourceBindsEveryCurrentSemanticExactly() ||
      !TemporalConfigurationsPrepareButResetRequiresReconstruction()) {
    return EXIT_FAILURE;
  }
  g_cuda = nullptr;
  std::cout << "Maxine production HAL owner tests passed\n";
  return EXIT_SUCCESS;
}
