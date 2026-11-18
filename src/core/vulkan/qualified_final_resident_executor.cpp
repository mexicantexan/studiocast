#include "core/vulkan/qualified_final_resident_executor.h"
#include "core/open_video/dlib_shape_predictor_data.h"
#include "core/open_video/prepared_vulkan_fastdvdnet.h"
#include "core/open_video/prepared_vulkan_tracking.h"
#include "core/vulkan/video_execution_factory.h"
#include "core/vulkan/video_execution_factory_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>

namespace studiocast::vulkan {
namespace {

using hal::BackendId;
using hal::ReasonId;
using hal::SemanticEffectId;
using hal::StatusId;

void SetReason(ReasonId *out, ReasonId value) noexcept {
  if (out)
    *out = value;
}

hal::ExecutionDomainId MakeDomain(std::uint64_t adapter_instance_id,
                                  const VulkanDevice &device) noexcept {
  hal::ExecutionDomainId domain;
  domain.backend = BackendId::vulkan;
  domain.api_family = hal::ApiFamily::vulkan;
  domain.physical_device.vendor_id = device.identity().vendor_id;
  domain.physical_device.device_id = device.identity().device_id;
  domain.physical_device.stable_id = device.identity().device_uuid;
  domain.logical_context_generation = device.context_identity().generation;
  domain.allocator_family = hal::AllocatorFamily::vulkan;
  domain.memory_handle_family = hal::MemoryHandleFamily::vulkan_buffer;
  domain.queue_or_stream_identity = static_cast<std::uint64_t>(
      reinterpret_cast<std::uintptr_t>(device.queue()));
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::vulkan_queue_fence;
  domain.adapter_instance_id = adapter_instance_id;
  return domain;
}

bool IsRgb(hal::PixelFormat format) noexcept {
  return format == hal::PixelFormat::rgb_u8 ||
         format == hal::PixelFormat::bgr_u8;
}

VulkanPixelFormat ToVulkan(hal::PixelFormat format) noexcept {
  return format == hal::PixelFormat::bgr_u8 ? VulkanPixelFormat::bgr_u8
                                            : VulkanPixelFormat::rgb_u8;
}

bool BudgetContains(const hal::ResourceBudget &limit,
                    const hal::ResourceBudget &used) noexcept {
  return used.persistent_bytes <= limit.persistent_bytes &&
         used.scratch_bytes <= limit.scratch_bytes &&
         used.upload_staging_bytes <= limit.upload_staging_bytes &&
         used.readback_staging_bytes <= limit.readback_staging_bytes &&
         used.resource_count <= limit.resource_count &&
         used.in_flight_slots <= limit.in_flight_slots;
}

bool Add(std::uint64_t *value, std::uint64_t amount) noexcept {
  if (!value || amount > std::numeric_limits<std::uint64_t>::max() - *value)
    return false;
  *value += amount;
  return true;
}

bool AllocateImage(VulkanDevice *device, VulkanImage *image, int width,
                   int height, VulkanPixelFormat format, bool mapped,
                   std::string *error) {
  return device && image &&
         image->Allocate(device, width, height, format, mapped, error);
}

bool MatchesStageSubset(const hal::PreparedProgram &program,
                        const VulkanResidentProgramOptions &options) noexcept {
  bool vignette = false;
  bool mirror = false;
  bool keylight = false;
  bool denoise = false, eye = false, auto_frame = false;
  SemanticEffectId background = SemanticEffectId::invalid;
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    const auto semantic = program.stages[i].semantic;
    if (semantic == SemanticEffectId::video_noise_removal) {
      if (denoise || eye || background != SemanticEffectId::invalid ||
          keylight || auto_frame || vignette || mirror)
        return false;
      denoise = true;
    } else if (semantic == SemanticEffectId::eye_contact) {
      if (eye || background != SemanticEffectId::invalid || keylight ||
          auto_frame || vignette || mirror)
        return false;
      eye = true;
    } else if (semantic == SemanticEffectId::auto_frame) {
      if (auto_frame || vignette || mirror)
        return false;
      auto_frame = true;
    } else if (semantic == SemanticEffectId::virtual_background_blur ||
               semantic == SemanticEffectId::virtual_background_remove ||
               semantic == SemanticEffectId::virtual_background_replace) {
      if (background != SemanticEffectId::invalid || keylight || auto_frame ||
          vignette || mirror)
        return false;
      background = semantic;
    } else if (semantic == SemanticEffectId::virtual_key_light) {
      if (keylight || auto_frame || vignette || mirror)
        return false;
      keylight = true;
    } else if (semantic == SemanticEffectId::vignette) {
      if (vignette || mirror)
        return false;
      vignette = true;
    } else if (semantic == SemanticEffectId::mirror) {
      if (mirror)
        return false;
      mirror = true;
    } else {
      return false;
    }
  }
  return vignette == options.request_vignette &&
         denoise == options.request_denoise && eye == options.request_eye &&
         auto_frame == options.request_auto_frame &&
         mirror == options.request_mirror &&
         keylight == options.request_keylight &&
         background == options.background;
}

} // namespace

struct QualifiedFinalResidentExecutor::ModelEffects {
  VulkanVideoExecutionFactoryConfiguration config;
  std::shared_ptr<open_vulkan::VulkanInferenceSharedQuota> quota;
  open_vulkan::PreparedVulkanFastDvdnet denoise;
  open_vulkan::PreparedVulkanTracking tracking;
  open_vulkan::NcnnVulkanInference face, left, right;
  open_video::DlibPredictorData predictor;
  VulkanImage denoise_input, denoise_output, face_input, eye_output;
  VulkanImage auto_input, auto_output, vignette_input, vignette_output, alpha;
  ResidentCommandBatch batch;
  bool denoise_ready = false, face_ready = false, eye_ready = false;
  bool tracking_ready = false, tracked_vignette_ready = false;
  bool reset_tracking = true;

  static open_vulkan::VulkanInferenceTensorBinding
  Binding(VulkanDevice *device, const VulkanTensor &tensor,
          std::uint32_t dims) {
    open_vulkan::VulkanInferenceTensorBinding b;
    auto &r = b.resource;
    r.ownership_domain = device;
    r.logical_device = reinterpret_cast<std::uintptr_t>(device->device());
    r.buffer = reinterpret_cast<std::uintptr_t>(tensor.buffer());
    r.byte_size = tensor.byte_size();
    r.n = 1;
    r.c = tensor.c();
    r.h = tensor.h();
    r.w = tensor.w();
    r.device_resident = tensor.Valid();
    r.device_local_memory = tensor.device_local();
    r.host_mapped = tensor.mapped() != nullptr;
    r.context_id = device->context_identity().context_id;
    r.context_generation = device->context_identity().generation;
    r.runtime_queue_ownership = true;
    b.dims = dims;
    b.w = tensor.w();
    b.h = tensor.h();
    b.c = tensor.c();
    // Tracking stores the two eye angles as two scalar channels. The reviewed
    // FLX graph consumes the same contiguous bytes as a rank-one vector.
    if (dims == 1) {
      b.w = tensor.w() * tensor.h() * tensor.c();
      b.h = 1;
      b.c = 1;
    }
    b.row_stride_bytes = static_cast<std::size_t>(b.w) * sizeof(float);
    b.plane_stride_bytes = b.row_stride_bytes * static_cast<std::size_t>(b.h);
    return b;
  }
  bool PrepareGraph(open_vulkan::NcnnVulkanInference &runtime,
                    const open_video::ModelPack &pack, const char *role,
                    const std::array<const VulkanTensor *, 3> &inputs,
                    const std::array<const VulkanTensor *, 12> &outputs,
                    std::string *error) {
    const auto found = std::find_if(
        pack.ncnn_vulkan_networks.begin(), pack.ncnn_vulkan_networks.end(),
        [&](const auto &network) { return network.role == role; });
    if (found == pack.ncnn_vulkan_networks.end()) {
      if (error)
        *error = "Required resident inference graph role is missing.";
      return false;
    }
    if (!open_video::ValidateProductionNcnnVulkanNetworkPack(pack, *found,
                                                             error))
      return false;
    auto *device = quota->device;
    open_vulkan::VulkanInferenceBindings bindings;
    bindings.owner_generation = config.factory_id;
    bindings.input_count = found->input_count;
    bindings.output_count = found->output_count;
    for (std::size_t i = 0; i < bindings.input_count; ++i) {
      if (!inputs[i])
        return false;
      bindings.inputs[i] = Binding(device, *inputs[i], found->inputs[i].dims);
    }
    for (std::size_t i = 0; i < bindings.output_count; ++i) {
      if (!outputs[i])
        return false;
      bindings.outputs[i] =
          Binding(device, *outputs[i], found->outputs[i].dims);
    }
    open_vulkan::VulkanMattingDeviceContext context;
    context.ownership_domain = device;
    context.physical_device =
        reinterpret_cast<std::uintptr_t>(device->physical_device());
    context.logical_device = reinterpret_cast<std::uintptr_t>(device->device());
    context.queue = reinterpret_cast<std::uintptr_t>(device->queue());
    context.queue_family_index = device->queue_family_index();
    context.context_id = device->context_identity().context_id;
    context.context_generation = device->context_identity().generation;
    context.vendor_id = device->identity().vendor_id;
    context.device_id = device->identity().device_id;
    context.stable_device_id = device->identity().stable_id;
    context.selected_device_index = device->diagnostics().selected_device_index;
    context.non_cpu_device_selected =
        device->diagnostics().non_cpu_device_selected;
    context.compute_queue_available =
        device->diagnostics().compute_queue_available;
    context.context_healthy = device->health_state().healthy();
    const auto prepared = runtime.Prepare(context, *found, bindings, quota);
    if (!prepared.ok()) {
      if (error)
        *error = prepared.detail;
      return false;
    }
    const auto warmed = runtime.Warmup();
    if (!warmed.ok() && error)
      *error = warmed.detail;
    return warmed.ok();
  }
};

bool QualifiedFinalResidentExecutor::InitializeModels(
    const VulkanVideoExecutionFactoryConfiguration &configuration,
    std::string *error) {
  if (!configuration.effects || !kernels_ || !kernels_->Initialized() ||
      configuration.input_width <= 0 || configuration.input_height <= 0 ||
      configuration.output_width <= 0 || configuration.output_height <= 0)
    return false;
  model_effects_ = std::make_unique<ModelEffects>();
  auto &m = *model_effects_;
  m.config = configuration;
  effects_ = *configuration.effects;
  auto *device = kernels_->device();
  m.quota = std::make_shared<open_vulkan::VulkanInferenceSharedQuota>();
  m.quota->device = device;
  m.quota->context = device->context_identity();
  m.quota->allocation_budget_bytes = configuration.reservation.persistent_bytes;
  for (const auto bytes : {configuration.reservation.scratch_bytes,
                           configuration.reservation.upload_staging_bytes,
                           configuration.reservation.readback_staging_bytes})
    if (!Add(&m.quota->allocation_budget_bytes, bytes))
      return false;
  m.quota->allocation_budget_bytes =
      std::min<std::size_t>(m.quota->allocation_budget_bytes,
                            device->allocation_stats().budget_bytes);
  m.quota->allocation_resource_limit = configuration.reservation.resource_count;
  if (!device->ConfigureAllocationLimits(m.quota->allocation_budget_bytes,
                                         m.quota->allocation_resource_limit,
                                         error) ||
      !m.batch.Initialize(device, kMaxResidentFrameCommands, error) ||
      !kernels_->PrepareDeferredSlotsForSetup(64, error))
    return false;
  const auto image = [&](VulkanImage &target, bool final = false,
                         VulkanPixelFormat format = VulkanPixelFormat::rgb_u8) {
    return target.Allocate(
        device, final ? configuration.output_width : configuration.input_width,
        final ? configuration.output_height : configuration.input_height,
        format, false, error);
  };
  if (configuration.denoise_pack) {
    initialization_failed_semantic_ = SemanticEffectId::video_noise_removal;
    if (!image(m.denoise_input) || !image(m.denoise_output))
      return false;
    const auto &networks = configuration.denoise_pack->ncnn_vulkan_networks;
    const auto main =
        std::find_if(networks.begin(), networks.end(), [](const auto &network) {
          return network.role == "main";
        });
    if (main == networks.end())
      return false;
    if (!open_video::ValidateProductionNcnnVulkanNetworkPack(
            *configuration.denoise_pack, *main, error))
      return false;
    open_vulkan::PreparedVulkanFastDvdnetSetup setup;
    setup.device = device;
    setup.domain = evidence_.execution_domain;
    setup.owner_generation = configuration.factory_id;
    setup.width = configuration.input_width;
    setup.height = configuration.input_height;
    setup.strength = effects_.video_noise_removal.strength;
    setup.graph = *main;
    setup.quota = m.quota;
    setup.reservation = configuration.reservation;
    if (!m.denoise.Prepare(setup, m.denoise_input, m.denoise_output, error))
      return false;
    m.denoise_ready = true;
    initialization_failed_semantic_ = SemanticEffectId::invalid;
  }
  const bool eye = effects_.eye_contact.enabled;
  const bool face = eye || effects_.auto_frame.enabled;
  const bool tracked = effects_.vignette.enabled &&
                       effects_.vignette.center_on_tracked_face &&
                       effects_.auto_frame.enabled;
  if (face || tracked) {
    initialization_failed_semantic_ = SemanticEffectId::auto_frame;
    if (!configuration.face_pack || !image(m.face_input) ||
        (eye && !image(m.eye_output)) ||
        (effects_.auto_frame.enabled &&
         (!image(m.auto_input) || !image(m.auto_output))) ||
        (tracked &&
         (!image(m.vignette_input, true) || !image(m.vignette_output, true))) ||
        (configuration.matting_pack &&
         !image(m.alpha, false, VulkanPixelFormat::f32_1)))
      return false;
    open_vulkan::PreparedVulkanTrackingSetup setup;
    setup.device = device;
    setup.domain = evidence_.execution_domain;
    setup.owner_generation = configuration.factory_id;
    setup.frame_width = configuration.input_width;
    setup.frame_height = configuration.input_height;
    setup.final_width = configuration.output_width;
    setup.final_height = configuration.output_height;
    setup.face_analysis = face;
    setup.eye_contact = eye;
    setup.auto_frame = effects_.auto_frame.enabled;
    setup.vignette = tracked;
    setup.eye_strength = effects_.eye_contact.strength;
    setup.look_away_enabled = effects_.eye_contact.look_away_enabled;
    setup.auto_frame_strength = effects_.auto_frame.strength;
    setup.auto_frame_smoothing = effects_.auto_frame.smoothing;
    setup.auto_frame_headroom = effects_.auto_frame.headroom;
    setup.vignette_intensity = effects_.vignette.intensity;
    setup.vignette_tracked_center = tracked;
    setup.matte_shared_current_frame =
        effects_.virtual_background.mode !=
            video::effects::VirtualBackgroundMode::none ||
        effects_.virtual_key_light.enabled;
    setup.reservation = configuration.reservation;
    const auto validated = [&](const open_video::ModelPack &pack,
                               const char *kind) {
      const auto file = std::find_if(
          pack.files.begin(), pack.files.end(),
          [&](const auto &artifact) { return artifact.kind == kind; });
      return file == pack.files.end()
                 ? open_video::ModelArtifactValidationResult{}
                 : open_video::ValidateModelArtifactIdentity(pack, *file);
    };
    const auto face_identity =
        validated(*configuration.face_pack, "ncnn_param");
    if (!face_identity.ok())
      return false;
    setup.face_artifact = face_identity.identity;
    if (eye) {
      initialization_failed_semantic_ = SemanticEffectId::eye_contact;
      if (!configuration.landmarks_pack || !configuration.eye_pack)
        return false;
      const auto landmark =
          validated(*configuration.landmarks_pack, "dlib_shape_predictor");
      if (!landmark.ok() ||
          !open_video::LoadDlibPredictorData(
              *configuration.landmarks_pack, landmark.identity,
              configuration.reservation.persistent_bytes, &m.predictor, error))
        return false;
      setup.predictor = &m.predictor;
      setup.landmark_artifact = landmark.identity;
      for (unsigned eye_index = 0; eye_index < 2; ++eye_index) {
        const auto &pack = *configuration.eye_pack;
        const auto network = std::find_if(
            pack.ncnn_vulkan_networks.begin(), pack.ncnn_vulkan_networks.end(),
            [&](const auto &item) {
              return item.role == (eye_index == 0 ? "left" : "right");
            });
        if (network == pack.ncnn_vulkan_networks.end())
          return false;
        const auto artifact = std::find_if(
            pack.files.begin(), pack.files.end(),
            [&](const auto &item) { return item.name == network->param_file; });
        if (artifact == pack.files.end())
          return false;
        const auto identity =
            open_video::ValidateModelArtifactIdentity(pack, *artifact);
        if (!identity.ok())
          return false;
        (eye_index == 0 ? setup.eye_left_artifact : setup.eye_right_artifact) =
            identity.identity;
      }
    }
    initialization_failed_semantic_ =
        eye ? SemanticEffectId::eye_contact : SemanticEffectId::auto_frame;
    if (!m.tracking.Prepare(setup, error))
      return false;
    m.tracking_ready = true;
    const std::array<const VulkanTensor *, 3> face_inputs{
        &m.tracking.FaceInput(), nullptr, nullptr};
    initialization_failed_semantic_ = SemanticEffectId::auto_frame;
    if (!m.PrepareGraph(m.face, *configuration.face_pack, "main", face_inputs,
                        m.tracking.FaceOutputs(), error))
      return false;
    m.face_ready = true;
    if (eye) {
      initialization_failed_semantic_ = SemanticEffectId::eye_contact;
      for (unsigned i = 0; i < 2; ++i) {
        const auto in = m.tracking.EyeInputs(i);
        const auto out = m.tracking.EyePrefixOutputs(i);
        const std::array<const VulkanTensor *, 3> inputs{in.rgb, in.anchors,
                                                         in.angles};
        std::array<const VulkanTensor *, 12> outputs{};
        outputs[0] = out.flow;
        outputs[1] = out.lcm;
        if (!m.PrepareGraph(i == 0 ? m.left : m.right, *configuration.eye_pack,
                            i == 0 ? "left" : "right", inputs, outputs, error))
          return false;
      }
      m.eye_ready = true;
    }
    // Record every enabled immutable phase against canonical fixed images at
    // setup. Abort proves no phase was executed; live calls only append them.
    initialization_failed_semantic_ = SemanticEffectId::auto_frame;
    bool recorded =
        m.batch.BeginFrame(1, error) &&
        m.tracking.AppendReset(m.batch, error) &&
        m.tracking.AppendFacePreprocess(m.batch, m.face_input, error) &&
        m.tracking.AppendFaceDecode(m.batch, error);
    if (recorded && eye) {
      initialization_failed_semantic_ = SemanticEffectId::eye_contact;
      recorded = m.tracking.AppendLandmarks(m.batch, m.face_input, error) &&
                 m.tracking.AppendEyeInputs(m.batch, m.face_input, error) &&
                 m.tracking.AppendEyeTailAndComposite(m.batch, m.face_input,
                                                      m.eye_output, error);
    }
    if (recorded && setup.auto_frame) {
      initialization_failed_semantic_ = SemanticEffectId::auto_frame;
      recorded = m.tracking.AppendAutoFrame(
          m.batch, m.auto_input, m.alpha.Valid() ? &m.alpha : nullptr,
          m.auto_output, error);
    }
    if (recorded && tracked) {
      initialization_failed_semantic_ = SemanticEffectId::vignette;
      recorded = m.tracking.AppendVignette(m.batch, m.vignette_input,
                                           m.vignette_output, error);
    }
    if (!recorded) {
      m.batch.AbortUnsubmitted();
      return false;
    }
    m.batch.AbortUnsubmitted();
    initialization_failed_semantic_ = SemanticEffectId::invalid;
    if (!m.tracking.SealBindingsForExecution(error))
      return false;
    m.tracked_vignette_ready = tracked;
  }
  initialization_failed_semantic_ = SemanticEffectId::invalid;
  RefreshEvidence();
  return true;
}

QualifiedFinalResidentExecutor::QualifiedFinalResidentExecutor(
    std::uint64_t adapter_instance_id,
    kernels::UtilityKernels &kernels) noexcept
    : adapter_instance_id_(adapter_instance_id), kernels_(&kernels) {
  RefreshEvidence();
}

QualifiedFinalResidentExecutor::~QualifiedFinalResidentExecutor() {
  Shutdown();
}

bool QualifiedFinalResidentExecutor::InitializeMatting(
    const open_video::ModelPack &pack,
    const video::effects::BroadcastCameraEffects &effects, int width,
    int height, std::string *error) {
  if (!kernels_ || !kernels_->Initialized() || !pack.matting || width <= 0 ||
      height <= 0)
    return false;
  effects_ = effects;
  if (!matting_alpha_.Allocate(kernels_->device(), pack.matting->output.width,
                               pack.matting->output.height,
                               VulkanPixelFormat::f32_1, false, error))
    return false;
  matting_ = std::make_unique<open_vulkan::OpenVulkanMattingSession>(
      kernels_->device(), kernels_, pack);
  if (!matting_->BindAlphaOutputForSetup(&matting_alpha_, error) ||
      !matting_->EnsureInitialized(width, height, error) ||
      !matting_->Warmup(error))
    return false;
  matting_readiness_ = matting_->Readiness();
  RefreshEvidence();
  return matting_readiness_.production_ready;
}

void QualifiedFinalResidentExecutor::RefreshEvidence() noexcept {
  evidence_ = {};
  evidence_.build_enabled = true;
  evidence_.blocker = ReasonId::capability_evidence_incomplete;
  if (!kernels_ || adapter_instance_id_ == 0 || !kernels_->Initialized() ||
      !kernels_->device())
    return;

  const auto &device = *kernels_->device();
  const auto &diagnostics = device.diagnostics();
  evidence_.execution_domain = MakeDomain(adapter_instance_id_, device);
  evidence_.runtime_loaded = diagnostics.runtime_library_found;
  evidence_.non_cpu_physical_device =
      diagnostics.non_cpu_device_selected && !diagnostics.cpu_device_selected;
  evidence_.compute_queue_available = diagnostics.compute_queue_available;
  evidence_.owned_context_healthy =
      diagnostics.context_healthy &&
      device.health().health == VulkanContextHealth::healthy;
  evidence_.utility_kernels_ready = kernels_->Initialized();
  evidence_.exact_device_and_queue_shared =
      device.identity().device_uuid_valid &&
      device.context_identity().Valid() && device.queue() != nullptr;
  evidence_.persistent_device_local_resources = true;
  evidence_.bounded_reusable_allocations = true;
  evidence_.no_host_intermediate_storage = true;
  evidence_.one_upload_transport = true;
  evidence_.one_final_readback_transport = true;
  const bool common =
      diagnostics.production_hardware_ready && evidence_.runtime_loaded &&
      evidence_.non_cpu_physical_device && evidence_.compute_queue_available &&
      evidence_.owned_context_healthy && evidence_.utility_kernels_ready &&
      evidence_.exact_device_and_queue_shared;
  evidence_.matting_production_ready =
      common && matting_ && matting_readiness_.production_ready;
  if (model_effects_) {
    evidence_.denoise_production_ready =
        common && model_effects_->denoise_ready;
    evidence_.face_production_ready = common && model_effects_->face_ready;
    evidence_.eye_production_ready = common && model_effects_->eye_ready;
    evidence_.tracking_gpu_ready =
        common && model_effects_->tracking_ready && effects_.auto_frame.enabled;
    evidence_.landmarks_production_ready = common && model_effects_->eye_ready;
    evidence_.shader_float64_enabled =
        common && device.shader_float64_enabled();
    evidence_.vignette_tracked_center_ready =
        common && model_effects_->tracked_vignette_ready;
  }
  evidence_.mirror_production_ready = common;
  evidence_.vignette_fixed_center_production_ready = common;
  evidence_.blocker =
      common ? ReasonId::none : ReasonId::capability_evidence_incomplete;
}

VulkanResidentCapabilityEvidence
QualifiedFinalResidentExecutor::CapabilityEvidence() const noexcept {
  return evidence_;
}

void QualifiedFinalResidentExecutor::CollectModelDiagnostics(
    VulkanProductionModelDiagnostics *out) const {
  if (!out)
    return;
  out->capabilities = evidence_;
  out->matting_readiness = matting_readiness_;
  if (matting_)
    out->matting = matting_->RuntimeEvidence();
  if (model_effects_) {
    out->denoise = model_effects_->denoise.evidence();
    out->face = model_effects_->face.Evidence();
    out->left_eye = model_effects_->left.Evidence();
    out->right_eye = model_effects_->right.Evidence();
    out->completed_batches = model_effects_->batch.completion_count();
  }
}

hal::StatusId QualifiedFinalResidentExecutor::Prepare(
    const hal::PreparedProgram &program,
    const VulkanResidentProgramOptions &options,
    VulkanExecutorPreparedProgram *prepared, ReasonId *reason) noexcept {
  if (prepared)
    *prepared = {};
  SetReason(reason, ReasonId::none);
  const auto fail = [&](StatusId status, ReasonId why) {
    SetReason(reason, why);
    return status;
  };
  if (shutdown_)
    return fail(StatusId::stopped, ReasonId::shutdown_requested);
  if (!prepared || !kernels_ || !kernels_->device())
    return fail(StatusId::blocked, ReasonId::capability_evidence_incomplete);
  RefreshEvidence();
  if (evidence_.blocker != ReasonId::none)
    return fail(StatusId::blocked, evidence_.blocker);
  if (program.execution_domain != evidence_.execution_domain ||
      program.owner_adapter_instance_id != adapter_instance_id_)
    return fail(StatusId::blocked, ReasonId::wrong_execution_domain);
  if (program.program_id == 0 || options.program_id != program.program_id ||
      options.canonical_plan_hash == 0 ||
      options.canonical_plan_hash !=
          program.invalidation_key.canonical_plan_hash ||
      !MatchesStageSubset(program, options) ||
      (options.request_vignette &&
       options.vignette_center !=
           VulkanVignetteCenterMode::fixed_frame_center &&
       !(model_effects_ && evidence_.vignette_tracked_center_ready &&
         options.vignette_center == VulkanVignetteCenterMode::tracked_face)) ||
      options.vignette_intensity_percent < 0 ||
      options.vignette_intensity_percent > 100 ||
      !IsRgb(program.host_input.format) || !IsRgb(program.host_output.format) ||
      program.host_input.format != program.host_output.format ||
      program.host_input.width == 0 || program.host_input.height == 0 ||
      program.host_output.width == 0 || program.host_output.height == 0) {
    return fail(StatusId::blocked, ReasonId::malformed_contract);
  }
  if (program.host_input.width >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      program.host_input.height >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      program.host_output.width >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      program.host_output.height >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return fail(StatusId::blocked, ReasonId::malformed_descriptor);
  }
  if (model_effects_) {
    const auto &c = model_effects_->config;
    if (program.host_input.width != static_cast<std::uint32_t>(c.input_width) ||
        program.host_input.height !=
            static_cast<std::uint32_t>(c.input_height) ||
        program.host_output.width !=
            static_cast<std::uint32_t>(c.output_width) ||
        program.host_output.height !=
            static_cast<std::uint32_t>(c.output_height) ||
        (options.request_denoise &&
         options.denoise_strength != effects_.video_noise_removal.strength) ||
        (options.request_eye &&
         (options.eye_strength != effects_.eye_contact.strength ||
          options.eye_look_away != effects_.eye_contact.look_away_enabled)) ||
        (options.request_auto_frame &&
         (options.auto_frame_strength != effects_.auto_frame.strength ||
          options.auto_frame_smoothing != effects_.auto_frame.smoothing ||
          options.auto_frame_headroom != effects_.auto_frame.headroom)) ||
        (options.vignette_center == VulkanVignetteCenterMode::tracked_face &&
         options.vignette_intensity_percent != effects_.vignette.intensity))
      return fail(StatusId::blocked, ReasonId::reconfiguration_required);
  }

  Slot *slot = nullptr;
  for (auto &candidate : slots_) {
    if (!candidate.occupied) {
      slot = &candidate;
      break;
    }
  }
  if (!slot)
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);

  slot->occupied = true;
  slot->state_id = next_state_id_++;
  if (slot->state_id == 0)
    slot->state_id = next_state_id_++;
  slot->program_id = program.program_id;
  slot->canonical_plan_hash = options.canonical_plan_hash;
  slot->execution_domain = program.execution_domain;
  slot->options = options;
  slot->input_width = program.host_input.width;
  slot->input_height = program.host_input.height;
  slot->output_width = program.host_output.width;
  slot->output_height = program.host_output.height;
  slot->format = ToVulkan(program.host_input.format);

  std::string error;
  auto *device = kernels_->device();
  const int iw = static_cast<int>(slot->input_width);
  const int ih = static_cast<int>(slot->input_height);
  const int ow = static_cast<int>(slot->output_width);
  const int oh = static_cast<int>(slot->output_height);
  bool ok = AllocateImage(device, &slot->upload_staging, iw, ih, slot->format,
                          true, &error) &&
            AllocateImage(device, &slot->source, iw, ih, slot->format, false,
                          &error) &&
            AllocateImage(device, &slot->resize_scratch, ow, oh, slot->format,
                          false, &error) &&
            AllocateImage(device, &slot->readback_staging, ow, oh, slot->format,
                          true, &error);
  const bool needs_matting = options.background != SemanticEffectId::invalid ||
                             options.request_keylight;
  if (ok && needs_matting) {
    ok = matting_ && matting_readiness_.production_ready &&
         AllocateImage(device, &slot->alpha, iw, ih, VulkanPixelFormat::f32_1,
                       false, &error) &&
         AllocateImage(device, &slot->alpha_tmp, iw, ih,
                       VulkanPixelFormat::f32_1, false, &error) &&
         AllocateImage(device, &slot->alpha_feathered, iw, ih,
                       VulkanPixelFormat::f32_1, false, &error);
    if (ok && options.background != SemanticEffectId::invalid) {
      ok = AllocateImage(device, &slot->background_output, iw, ih, slot->format,
                         false, &error);
      if (ok && options.background == SemanticEffectId::virtual_background_blur)
        ok = AllocateImage(device, &slot->blur_tmp, iw, ih, slot->format, false,
                           &error) &&
             AllocateImage(device, &slot->blurred, iw, ih, slot->format, false,
                           &error) &&
             slot->background_blur.EnsureInitialized(
                 kernels_, iw, ih, matting_readiness_, &error);
      if (ok &&
          options.background == SemanticEffectId::virtual_background_remove) {
        char color[8];
        std::snprintf(color, sizeof(color), "#%06x", options.remove_color_rgb);
        ok = slot->background_remove.EnsureInitialized(
            kernels_, iw, ih, options.background_strength, color,
            matting_readiness_, &error);
      }
      if (ok &&
          options.background == SemanticEffectId::virtual_background_replace) {
        video::detail::PreparedReplaceBackgroundSource asset;
        ok = video::detail::PrepareReplaceBackgroundSourceForConfig(
                 effects_.virtual_background.replace_path, &asset, &error) &&
             slot->background_replace.EnsureInitialized(
                 kernels_, iw, ih, options.background_strength, asset,
                 matting_readiness_, &slot->replace_counters, &error);
      }
    }
    if (ok && options.request_keylight)
      ok = AllocateImage(device, &slot->keylight_output, iw, ih, slot->format,
                         false, &error) &&
           slot->keylight.EnsureInitialized(
               kernels_, iw, ih, options.keylight_intensity,
               options.keylight_temperature, options.keylight_pan,
               matting_readiness_, &error);
  }
  if (ok && options.request_vignette) {
    ok = AllocateImage(device, &slot->vignette_output, ow, oh, slot->format,
                       false, &error) &&
         slot->vignette.EnsureInitialized(kernels_, ow, oh,
                                          options.vignette_intensity_percent,
                                          &slot->vignette_counters, &error);
  }
  if (ok && options.request_mirror) {
    ok = AllocateImage(device, &slot->mirror_output, ow, oh, slot->format,
                       false, &error) &&
         slot->mirror.EnsureInitialized(kernels_, ow, oh, &error);
  }
  if (!ok) {
    Release(slot);
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);
  }

  hal::ResourceBudget used;
  used.in_flight_slots = 1;
  used.resource_count = 4;
  bool budget_valid =
      Add(&used.upload_staging_bytes, slot->upload_staging.byte_size()) &&
      Add(&used.readback_staging_bytes, slot->readback_staging.byte_size()) &&
      Add(&used.persistent_bytes, slot->source.byte_size()) &&
      Add(&used.scratch_bytes, slot->resize_scratch.byte_size());
  if (options.request_vignette) {
    budget_valid =
        budget_valid &&
        Add(&used.persistent_bytes, slot->vignette_output.byte_size()) &&
        Add(&used.persistent_bytes,
            slot->vignette.attenuation_factor_mask().byte_size());
    used.resource_count += 2;
  }
  if (options.request_mirror) {
    budget_valid = budget_valid &&
                   Add(&used.persistent_bytes, slot->mirror_output.byte_size());
    ++used.resource_count;
  }
  if (needs_matting) {
    // The graph's weights and retained activation arena are one fixed
    // reservation.
    budget_valid =
        budget_valid &&
        Add(&used.persistent_bytes, UINT64_C(256) * 1024 * 1024) &&
        Add(&used.persistent_bytes, matting_alpha_.byte_size()) &&
        Add(&used.persistent_bytes,
            static_cast<std::uint64_t>(matting_->pack().matting->input.width) *
                static_cast<std::uint64_t>(
                    matting_->pack().matting->input.height) *
                12u);
    used.resource_count += 514;
    for (const auto *image :
         {&slot->alpha, &slot->alpha_tmp, &slot->alpha_feathered,
          &slot->blur_tmp, &slot->blurred, &slot->background_output,
          &slot->keylight_output}) {
      if (image->Valid()) {
        budget_valid =
            budget_valid && Add(&used.persistent_bytes, image->byte_size());
        ++used.resource_count;
      }
    }
    if (slot->background_replace.initialized()) {
      for (const auto *image :
           {&slot->background_replace.source_image(),
            &slot->background_replace.replacement_image()}) {
        budget_valid =
            budget_valid && Add(&used.persistent_bytes, image->byte_size());
        ++used.resource_count;
      }
      budget_valid =
          budget_valid &&
          Add(&used.upload_staging_bytes,
              slot->background_replace.upload_staging_image().byte_size());
      ++used.resource_count;
    }
  }
  if (model_effects_) {
    const auto actual = device->allocation_stats();
    const auto separate = used.upload_staging_bytes +
                          used.readback_staging_bytes + used.scratch_bytes;
    if (actual.current_bytes >= separate)
      used.persistent_bytes = std::max<std::uint64_t>(
          used.persistent_bytes, actual.current_bytes - separate);
    if (actual.allocation_count > UINT32_MAX)
      budget_valid = false;
    else
      used.resource_count =
          std::max(used.resource_count,
                   static_cast<std::uint32_t>(actual.allocation_count));
  }
  if (!budget_valid || !BudgetContains(program.budget, used)) {
    Release(slot);
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);
  }
  slot->reserved = used;
  prepared->state_id = slot->state_id;
  prepared->reserved_budget = used;
  prepared->opaque = {slot->program_id, slot->canonical_plan_hash,
                      slot->execution_domain.logical_context_generation,
                      adapter_instance_id_};
  return StatusId::ready;
}

VulkanExecutorResult QualifiedFinalResidentExecutor::Execute(
    const VulkanExecutorPreparedProgram &prepared,
    const hal::PreparedProgram &program,
    const VulkanResidentProgramOptions &options,
    const VulkanHostFrameBinding &binding,
    std::uint64_t capture_sequence) noexcept {
  VulkanExecutorResult result;
  const auto started = std::chrono::steady_clock::now();
  const auto finish = [&]() {
    result.frame_elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    return result;
  };
  if (shutdown_) {
    result.status = StatusId::stopped;
    result.reason = ReasonId::shutdown_requested;
    return finish();
  }
  Slot *slot = Find(prepared.state_id);
  if (!slot || prepared.opaque[0] != slot->program_id ||
      prepared.opaque[1] != slot->canonical_plan_hash ||
      prepared.opaque[2] != slot->execution_domain.logical_context_generation ||
      prepared.opaque[3] != adapter_instance_id_) {
    result.status = StatusId::stale;
    result.reason = ReasonId::stale_program;
    return finish();
  }
  if (slot->quarantined) {
    result.status = StatusId::quarantined;
    result.reason = slot->quarantine_reason;
    result.device_lost = result.reason == ReasonId::device_lost;
    return finish();
  }
  const std::size_t minimum_input_row =
      static_cast<std::size_t>(slot->input_width) * 3u;
  const std::size_t minimum_output_row =
      static_cast<std::size_t>(slot->output_width) * 3u;
  const bool binding_valid =
      binding.input && binding.output &&
      binding.input_row_stride_bytes >= minimum_input_row &&
      binding.output_row_stride_bytes >= minimum_output_row &&
      binding.input_size_bytes >=
          binding.input_row_stride_bytes * slot->input_height &&
      binding.output_size_bytes >=
          binding.output_row_stride_bytes * slot->output_height;
  if (program.execution_domain != slot->execution_domain) {
    result.status = StatusId::blocked;
    result.reason = ReasonId::wrong_execution_domain;
    return finish();
  }
  if (capture_sequence == 0 || program.program_id != slot->program_id ||
      options.program_id != slot->program_id ||
      options.canonical_plan_hash != slot->canonical_plan_hash ||
      (options.request_vignette && !slot->options.request_vignette) ||
      (options.request_mirror && !slot->options.request_mirror) ||
      (options.request_denoise &&
       (!slot->options.request_denoise ||
        options.denoise_strength != slot->options.denoise_strength)) ||
      (options.request_eye &&
       (!slot->options.request_eye ||
        options.eye_strength != slot->options.eye_strength ||
        options.eye_look_away != slot->options.eye_look_away)) ||
      (options.request_auto_frame &&
       (!slot->options.request_auto_frame ||
        options.auto_frame_strength != slot->options.auto_frame_strength ||
        options.auto_frame_smoothing != slot->options.auto_frame_smoothing ||
        options.auto_frame_headroom != slot->options.auto_frame_headroom)) ||
      (options.request_vignette &&
       (options.vignette_center != slot->options.vignette_center ||
        options.vignette_intensity_percent !=
            slot->options.vignette_intensity_percent)) ||
      (options.background != SemanticEffectId::invalid &&
       (options.background != slot->options.background ||
        options.background_strength != slot->options.background_strength ||
        options.remove_color_rgb != slot->options.remove_color_rgb)) ||
      (options.request_keylight &&
       (!slot->options.request_keylight ||
        options.keylight_intensity != slot->options.keylight_intensity ||
        options.keylight_temperature != slot->options.keylight_temperature ||
        options.keylight_pan != slot->options.keylight_pan)) ||
      !MatchesStageSubset(program, options) || !binding_valid) {
    result.status = StatusId::blocked;
    result.reason = ReasonId::malformed_contract;
    return finish();
  }
  if (model_effects_)
    return ExecuteExpanded(slot, program, options, binding, capture_sequence);
  const std::uint64_t submissions_before =
      kernels_->synchronous_submission_count();
  if (!kernels_->UploadU8x3ToDeviceLocal(
          binding.input, binding.input_row_stride_bytes, slot->upload_staging,
          slot->source, nullptr)) {
    const auto failed_health = kernels_->device()->health_state();
    result.device_lost =
        failed_health.health == VulkanContextHealth::device_lost;
    slot->quarantined = !failed_health.healthy();
    result.status =
        slot->quarantined ? StatusId::quarantined : StatusId::failed;
    result.reason = result.device_lost ? ReasonId::device_lost
                                       : ReasonId::synchronization_failed;
    if (slot->quarantined)
      slot->quarantine_reason = result.reason;
    return finish();
  }
  result.transfers.uploads = 1;

  const VulkanImage *current = &slot->source;
  result.stage_count = program.stage_count;
  const bool needs_matting = options.background != SemanticEffectId::invalid ||
                             options.request_keylight;
  bool matte_ready = false;
  const auto runtime_before = matting_readiness_.runtime.inference_count;
  const auto resize_before = slot->alpha_resize_completions;
  const auto runtime_completions_before =
      matting_readiness_.runtime.completion_count;
  if (needs_matting && matting_) {
    matte_ready = matting_->Run(slot->source, &matting_alpha_, nullptr);
    matting_->UpdateCachedReadiness(&matting_readiness_);
    if (matte_ready) {
      matte_ready =
          kernels_->ResizeBilinearF32_1(matting_alpha_, slot->alpha, nullptr);
      if (matte_ready)
        ++slot->alpha_resize_completions;
    }
  }
  bool background_applied = false;
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    const auto semantic = program.stages[i].semantic;
    if (semantic == SemanticEffectId::mirror ||
        semantic == SemanticEffectId::vignette)
      continue;
    const auto stage_start = std::chrono::steady_clock::now();
    bool ok = false;
    if (matte_ready && semantic == SemanticEffectId::virtual_background_blur) {
      video::OpenVulkanVirtualBackgroundBlurInput in;
      in.foreground = current;
      in.alpha = &slot->alpha;
      in.alpha_tmp = &slot->alpha_tmp;
      in.alpha_feathered = &slot->alpha_feathered;
      in.blur_tmp = &slot->blur_tmp;
      in.blurred = &slot->blurred;
      in.output = &slot->background_output;
      in.strength = options.background_strength;
      in.capture_sequence = capture_sequence;
      in.resident_alpha_sequence = capture_sequence;
      in.alpha_resize_completion_count = slot->alpha_resize_completions;
      in.matting_readiness = &matting_readiness_;
      ok = slot->background_blur.Apply(in, &slot->blur_counters, nullptr);
    } else if (matte_ready &&
               semantic == SemanticEffectId::virtual_background_remove) {
      video::OpenVulkanVirtualBackgroundRemoveInput in;
      in.foreground = current;
      in.alpha = &slot->alpha;
      in.alpha_tmp = &slot->alpha_tmp;
      in.alpha_feathered = &slot->alpha_feathered;
      in.output = &slot->background_output;
      in.capture_sequence = capture_sequence;
      in.resident_alpha_sequence = capture_sequence;
      in.alpha_resize_completion_count = slot->alpha_resize_completions;
      in.matting_readiness = &matting_readiness_;
      ok = slot->background_remove.Apply(in, &slot->remove_counters, nullptr);
    } else if (matte_ready &&
               semantic == SemanticEffectId::virtual_background_replace) {
      video::OpenVulkanVirtualBackgroundReplaceInput in;
      in.foreground = current;
      in.alpha = &slot->alpha;
      in.alpha_tmp = &slot->alpha_tmp;
      in.alpha_feathered = &slot->alpha_feathered;
      in.output = &slot->background_output;
      in.capture_sequence = capture_sequence;
      in.resident_alpha_sequence = capture_sequence;
      in.alpha_resize_completion_count = slot->alpha_resize_completions;
      in.matting_readiness = &matting_readiness_;
      ok = slot->background_replace.Apply(in, &slot->replace_counters, nullptr);
    } else if (matte_ready && semantic == SemanticEffectId::virtual_key_light) {
      video::OpenVulkanVirtualKeyLightInput in;
      in.foreground = current;
      in.alpha = &slot->alpha;
      in.output = &slot->keylight_output;
      in.capture_sequence = capture_sequence;
      in.resident_alpha_sequence = capture_sequence;
      in.alpha_resize_completion_count_before =
          background_applied ? slot->alpha_resize_completions : resize_before;
      in.alpha_resize_completion_count_after = slot->alpha_resize_completions;
      in.matting_inference_count_before =
          background_applied ? matting_readiness_.runtime.inference_count
                             : runtime_before;
      in.matte_source =
          background_applied
              ? video::OpenVulkanVirtualKeyLightMatteSource::reused_same_frame
              : video::OpenVulkanVirtualKeyLightMatteSource::
                    independently_inferred;
      in.matting_readiness = &matting_readiness_;
      ok = slot->keylight.Apply(in, &slot->keylight_counters, nullptr);
      if (ok && !slot->keylight.parameters().passthrough())
        current = &slot->keylight_output;
    }
    if (ok && semantic != SemanticEffectId::virtual_key_light) {
      current = &slot->background_output;
      background_applied = true;
    }
    result.stages[i].succeeded = ok;
    result.stages[i].reason =
        ok ? ReasonId::none : ReasonId::optional_stage_failed_open;
    result.stages[i].elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - stage_start)
            .count());
    if (!ok && program.stages[i].required) {
      result.status = StatusId::failed;
      result.reason = ReasonId::required_stage_failed;
      result.transfers.synchronizations =
          kernels_->synchronous_submission_count() - submissions_before +
          matting_readiness_.runtime.completion_count -
          runtime_completions_before;
      return finish();
    }
  }

  video::OpenVulkanFinalResidentStageResources resources;
  resources.kernels = kernels_;
  resources.resize_scratch = &slot->resize_scratch;
  resources.vignette_output = &slot->vignette_output;
  resources.mirror_output = &slot->mirror_output;
  resources.vignette = &slot->vignette;
  resources.vignette_counters = &slot->vignette_counters;
  resources.mirror = &slot->mirror;
  resources.mirror_counters = &slot->mirror_counters;
  resources.require_prepared_resources = true;

  video::OpenVulkanFinalResidentStageInput input;
  input.source = current;
  input.output_width = static_cast<int>(slot->output_width);
  input.output_height = static_cast<int>(slot->output_height);
  input.request_fixed_center_vignette = options.request_vignette;
  input.request_mirror = options.request_mirror;
  input.vignette_intensity_percent = options.vignette_intensity_percent;
  input.preceding_effects_complete = true;
  input.unmirrored_analysis_complete = true;
  video::OpenVulkanFinalResidentStageResult stage_result;
  const bool have_output = video::ExecuteOpenVulkanFinalResidentStage(
      input, resources, &slot->final_counters, &stage_result);

  result.stage_count = program.stage_count;
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    if (program.stages[i].semantic != SemanticEffectId::vignette &&
        program.stages[i].semantic != SemanticEffectId::mirror)
      continue;
    const bool succeeded =
        program.stages[i].semantic == SemanticEffectId::vignette
            ? stage_result.vignette_applied
            : stage_result.mirror_applied;
    result.stages[i].succeeded = succeeded;
    result.stages[i].reason =
        succeeded ? ReasonId::none : ReasonId::optional_stage_failed_open;
    result.stages[i].elapsed_ns =
        program.stages[i].semantic == SemanticEffectId::vignette
            ? stage_result.vignette_elapsed_ns
            : stage_result.mirror_elapsed_ns;
  }
  if (!have_output || !stage_result.output ||
      !kernels_->ReadbackU8x3(*stage_result.output, slot->readback_staging,
                              binding.output, binding.output_row_stride_bytes,
                              nullptr)) {
    const auto failed_health = kernels_->device()->health_state();
    result.device_lost =
        failed_health.health == VulkanContextHealth::device_lost;
    slot->quarantined = !failed_health.healthy();
    result.status =
        slot->quarantined ? StatusId::quarantined : StatusId::failed;
    result.reason = result.device_lost  ? ReasonId::device_lost
                    : slot->quarantined ? ReasonId::synchronization_failed
                                        : ReasonId::output_write_failed;
    if (slot->quarantined)
      slot->quarantine_reason = result.reason;
    result.transfers.synchronizations =
        kernels_->synchronous_submission_count() - submissions_before +
        matting_readiness_.runtime.completion_count -
        runtime_completions_before;
    return finish();
  }
  result.transfers.final_readbacks = 1;
  result.transfers.synchronizations =
      kernels_->synchronous_submission_count() - submissions_before +
      matting_readiness_.runtime.completion_count - runtime_completions_before;
  result.output_written = true;
  // Transport completion is separate from per-stage policy. The common
  // adapter derives required failure or optional degradation while retaining
  // this successfully written output.
  result.status = StatusId::executed;
  result.reason = ReasonId::none;
  return finish();
}

VulkanExecutorResult QualifiedFinalResidentExecutor::ExecuteExpanded(
    Slot *slot, const hal::PreparedProgram &program,
    const VulkanResidentProgramOptions &options,
    const VulkanHostFrameBinding &binding, std::uint64_t sequence) noexcept {
  auto &m = *model_effects_;
  VulkanExecutorResult result;
  result.stage_count = program.stage_count;
  const auto started = std::chrono::steady_clock::now();
  const auto finish = [&]() {
    if (m.batch.state() == ResidentCommandBatchState::recording)
      m.batch.AbortUnsubmitted();
    kernels_->DiscardUnsubmittedOperation();
    kernels_->EndDeferredFrame();
    if (m.batch.state() == ResidentCommandBatchState::unsafe_pending ||
        !kernels_->device()->health_state().healthy()) {
      slot->quarantined = true;
      result.device_lost = kernels_->device()->health_state().health ==
                           VulkanContextHealth::device_lost;
      result.status = StatusId::quarantined;
      result.reason = result.device_lost ? ReasonId::device_lost
                                         : ReasonId::synchronization_failed;
      slot->quarantine_reason = result.reason;
    }
    result.frame_elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    return result;
  };
  if (!m.batch.BeginFrame(sequence, nullptr) ||
      !kernels_->BeginDeferredFrame(m.batch, nullptr)) {
    result.reason = ReasonId::synchronization_failed;
    return finish();
  }
  if (!kernels_->UploadU8x3ToDeviceLocal(
          binding.input, binding.input_row_stride_bytes, slot->upload_staging,
          slot->source, nullptr)) {
    result.reason = ReasonId::synchronization_failed;
    return finish();
  }
  const VulkanImage *current = &slot->source;
  bool face_queued = false, matte_queued = false, auto_queued = false;
  bool reset_queued = false;
  const VulkanImage &alpha = m.alpha.Valid() ? m.alpha : slot->alpha;
  const auto ensure_face = [&]() {
    if (face_queued)
      return true;
    if (!m.face_ready || !m.tracking_ready)
      return false;
    if (m.reset_tracking && !reset_queued) {
      if (!m.tracking.AppendReset(m.batch, nullptr))
        return false;
      reset_queued = true;
    }
    if (!kernels_->ResizeBilinear(*current, m.face_input, nullptr) ||
        !m.tracking.AppendFacePreprocess(m.batch, m.face_input, nullptr) ||
        !m.face.Enqueue(m.batch).ok() ||
        !m.tracking.AppendFaceDecode(m.batch, nullptr))
      return false;
    face_queued = true;
    return true;
  };
  const auto ensure_matte = [&]() {
    if (matte_queued)
      return true;
    if (!matting_ || !matting_readiness_.production_ready ||
        !matting_->Enqueue(*current, &matting_alpha_, m.batch, nullptr) ||
        !kernels_->ResizeBilinearF32_1(matting_alpha_, alpha, nullptr))
      return false;
    matte_queued = true;
    return true;
  };
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    const auto semantic = program.stages[i].semantic;
    if (semantic == SemanticEffectId::vignette ||
        semantic == SemanticEffectId::mirror)
      continue;
    const auto mark = m.batch.Mark();
    const bool prior_face = face_queued, prior_matte = matte_queued,
               prior_reset = reset_queued;
    const VulkanImage *next = current;
    bool ok = false;
    if (semantic == SemanticEffectId::video_noise_removal) {
      ok = options.denoise_strength == 0 ||
           (m.denoise_ready &&
            kernels_->ResizeBilinear(*current, m.denoise_input, nullptr) &&
            m.denoise.Append(m.batch, sequence, nullptr));
      if (ok && options.denoise_strength > 0)
        next = &m.denoise_output;
    } else if (semantic == SemanticEffectId::eye_contact) {
      ok = options.eye_strength == 0 ||
           (m.eye_ready && ensure_face() &&
            m.tracking.AppendLandmarks(m.batch, m.face_input, nullptr) &&
            m.tracking.AppendEyeInputs(m.batch, m.face_input, nullptr) &&
            m.left.Enqueue(m.batch).ok() && m.right.Enqueue(m.batch).ok() &&
            m.tracking.AppendEyeTailAndComposite(m.batch, m.face_input,
                                                 m.eye_output, nullptr));
      if (ok && options.eye_strength > 0)
        next = &m.eye_output;
    } else if (semantic == SemanticEffectId::auto_frame) {
      ok = ensure_face();
      bool current_matte = false;
      if (ok && matting_) {
        const auto matte_mark = m.batch.Mark();
        current_matte = ensure_matte();
        if (!current_matte) {
          kernels_->DiscardUnsubmittedOperation();
          ok = m.batch.Rollback(matte_mark, nullptr);
        }
      }
      ok = ok && kernels_->ResizeBilinear(*current, m.auto_input, nullptr) &&
           m.tracking.AppendAutoFrame(m.batch, m.auto_input,
                                      m.alpha.Valid() ? &m.alpha : nullptr,
                                      m.auto_output, nullptr, current_matte);
      if (ok) {
        next = &m.auto_output;
        auto_queued = true;
      }
    } else if (semantic == SemanticEffectId::virtual_key_light) {
      const auto &p = slot->keylight.parameters();
      ok = ensure_matte() &&
           (p.passthrough() ||
            kernels_->ApplyKeyLightU8x3(*current, alpha, p.target_r, p.target_g,
                                        p.target_b, p.intensity, p.direction,
                                        slot->keylight_output, nullptr));
      if (ok && !p.passthrough())
        next = &slot->keylight_output;
    } else if (semantic == SemanticEffectId::virtual_background_blur ||
               semantic == SemanticEffectId::virtual_background_remove ||
               semantic == SemanticEffectId::virtual_background_replace) {
      ok = ensure_matte();
      int feather = 0;
      if (semantic == SemanticEffectId::virtual_background_blur)
        feather = video::ResolveOpenVulkanVirtualBackgroundBlurParameters(
                      options.background_strength)
                      .alpha_feather_radius;
      else if (semantic == SemanticEffectId::virtual_background_remove)
        feather = slot->background_remove.parameters().alpha_feather_radius;
      else
        feather = slot->background_replace.parameters().alpha_feather_radius;
      const VulkanImage *composite_alpha = &alpha;
      if (ok && feather > 0) {
        ok = kernels_->BoxBlurSeparableF32_1(
            alpha, slot->alpha_tmp, slot->alpha_feathered, feather, nullptr);
        composite_alpha = &slot->alpha_feathered;
      }
      if (ok && semantic == SemanticEffectId::virtual_background_blur) {
        const auto p = video::ResolveOpenVulkanVirtualBackgroundBlurParameters(
            options.background_strength);
        ok = kernels_->BoxBlurCompositeAlphaU8x3(
            *current, slot->blur_tmp, slot->blurred, *composite_alpha,
            slot->background_output, p.background_radius, nullptr);
      } else if (ok &&
                 semantic == SemanticEffectId::virtual_background_remove) {
        const auto &p = slot->background_remove.parameters();
        ok = kernels_->CompositeAlphaSolidU8x3(
            *current, *composite_alpha, p.background_r, p.background_g,
            p.background_b, slot->background_output, nullptr);
      } else if (ok) {
        ok = kernels_->CompositeAlphaU8x3(
            *current, slot->background_replace.replacement_image(),
            *composite_alpha, slot->background_output, nullptr);
      }
      if (ok)
        next = &slot->background_output;
    }
    if (!ok) {
      kernels_->DiscardUnsubmittedOperation();
      if (!m.batch.Rollback(mark, nullptr)) {
        result.reason = ReasonId::synchronization_failed;
        return finish();
      }
      face_queued = prior_face;
      matte_queued = prior_matte;
      reset_queued = prior_reset;
      result.stages[i].reason = ReasonId::optional_stage_failed_open;
      if (program.stages[i].required) {
        result.reason = ReasonId::required_stage_failed;
        return finish();
      }
    } else {
      current = next;
      result.stages[i].succeeded = true;
    }
  }
  if (current->width() != static_cast<int>(slot->output_width) ||
      current->height() != static_cast<int>(slot->output_height)) {
    if (!kernels_->ResizeBilinear(*current, slot->resize_scratch, nullptr)) {
      result.reason = ReasonId::synchronization_failed;
      return finish();
    }
    current = &slot->resize_scratch;
  }
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    const auto semantic = program.stages[i].semantic;
    if (semantic != SemanticEffectId::vignette &&
        semantic != SemanticEffectId::mirror)
      continue;
    const auto mark = m.batch.Mark();
    bool ok = false;
    const VulkanImage *next = current;
    if (semantic == SemanticEffectId::vignette) {
      if (options.vignette_center == VulkanVignetteCenterMode::tracked_face &&
          auto_queued) {
        ok = kernels_->ResizeBilinear(*current, m.vignette_input, nullptr) &&
             m.tracking.AppendVignette(m.batch, m.vignette_input,
                                       m.vignette_output, nullptr);
        next = &m.vignette_output;
      } else {
        ok = kernels_->ApplyFinalVignetteU8x3(
            *current, nullptr, slot->vignette.attenuation_factor_mask(),
            slot->vignette_output, nullptr, nullptr);
        next = &slot->vignette_output;
      }
    } else {
      ok = kernels_->MirrorHorizontalU8x3(*current, slot->mirror_output,
                                          nullptr);
      next = &slot->mirror_output;
    }
    if (ok) {
      current = next;
      result.stages[i].succeeded = true;
    } else {
      kernels_->DiscardUnsubmittedOperation();
      if (!m.batch.Rollback(mark, nullptr) || program.stages[i].required) {
        result.reason = ReasonId::required_stage_failed;
        return finish();
      }
      result.stages[i].reason = ReasonId::optional_stage_failed_open;
    }
  }
  if (!kernels_->ReadbackU8x3(*current, slot->readback_staging, binding.output,
                              binding.output_row_stride_bytes, nullptr)) {
    result.reason = ReasonId::synchronization_failed;
    return finish();
  }
  result.transfers.uploads = 1;
  result.transfers.final_readbacks = 1;
  result.transfers.synchronizations = 1;
  if (matte_queued) {
    matting_->UpdateCachedReadiness(&matting_readiness_);
    ++slot->alpha_resize_completions;
  }
  if (reset_queued)
    m.reset_tracking = false;
  result.output_written = true;
  result.status = StatusId::executed;
  result.reason = ReasonId::none;
  return finish();
}

QualifiedFinalResidentExecutor::Slot *
QualifiedFinalResidentExecutor::Find(std::uint64_t state_id) noexcept {
  if (state_id == 0)
    return nullptr;
  for (auto &slot : slots_) {
    if (slot.occupied && slot.state_id == state_id)
      return &slot;
  }
  return nullptr;
}

hal::StatusId QualifiedFinalResidentExecutor::ResetTemporalState(
    const VulkanExecutorPreparedProgram &prepared, hal::ReasonId) noexcept {
  if (shutdown_)
    return StatusId::stopped;
  Slot *slot = Find(prepared.state_id);
  if (!slot || prepared.opaque[0] != slot->program_id ||
      prepared.opaque[1] != slot->canonical_plan_hash ||
      prepared.opaque[2] != slot->execution_domain.logical_context_generation ||
      prepared.opaque[3] != adapter_instance_id_)
    return StatusId::stale;
  if (slot->quarantined || !kernels_->device()->health_state().healthy())
    return StatusId::quarantined;
  if (model_effects_) {
    auto &m = *model_effects_;
    if (m.batch.state() == ResidentCommandBatchState::recording ||
        m.batch.state() == ResidentCommandBatchState::unsafe_pending ||
        !m.denoise.ResetTemporalState())
      return StatusId::quarantined;
    m.reset_tracking = true;
  }
  // Model-free/matting-only programs have no history. Model leaves extend
  // this exact forwarding boundary with their prepared reset contracts.
  return StatusId::ready;
}

void QualifiedFinalResidentExecutor::Release(Slot *slot) noexcept {
  if (!slot)
    return;
  slot->background_blur.Shutdown();
  slot->background_remove.Shutdown();
  slot->background_replace.Shutdown();
  slot->keylight.Shutdown();
  for (auto *image :
       {&slot->alpha, &slot->alpha_tmp, &slot->alpha_feathered, &slot->blur_tmp,
        &slot->blurred, &slot->background_output, &slot->keylight_output})
    image->Free();
  slot->alpha_resize_completions = 0;
  slot->vignette.Shutdown();
  slot->readback_staging.Free();
  slot->mirror_output.Free();
  slot->vignette_output.Free();
  slot->resize_scratch.Free();
  slot->source.Free();
  slot->upload_staging.Free();
  slot->occupied = false;
  slot->quarantined = false;
  slot->quarantine_reason = ReasonId::none;
  slot->state_id = 0;
  slot->program_id = 0;
  slot->canonical_plan_hash = 0;
  slot->execution_domain = {};
  slot->options = {};
  slot->input_width = slot->input_height = 0;
  slot->output_width = slot->output_height = 0;
  slot->vignette_counters = {};
  slot->mirror_counters = {};
  slot->final_counters = {};
  slot->reserved = {};
}

void QualifiedFinalResidentExecutor::DestroyPrepared(
    VulkanExecutorPreparedProgram *prepared) noexcept {
  if (!prepared)
    return;
  if (Slot *slot = Find(prepared->state_id)) {
    if (kernels_ && kernels_->device() &&
        !kernels_->device()->health_state().healthy()) {
      slot->quarantined = true;
      slot->quarantine_reason = kernels_->device()->health_state().health ==
                                        VulkanContextHealth::device_lost
                                    ? ReasonId::device_lost
                                    : ReasonId::synchronization_failed;
      return;
    }
    Release(slot);
  }
  *prepared = {};
}

void QualifiedFinalResidentExecutor::Shutdown() noexcept {
  (void)TryShutdown();
}

bool QualifiedFinalResidentExecutor::TryShutdown() noexcept {
  if (shutdown_)
    return true;
  if (!RetireFailedFrameAfterQuiescence())
    return false;
  if (model_effects_) {
    auto &m = *model_effects_;
    if (!m.denoise.Shutdown(nullptr) || !m.tracking.Shutdown(nullptr) ||
        !m.face.Shutdown() || !m.left.Shutdown() || !m.right.Shutdown())
      return false;
    model_effects_.reset();
  }
  for (auto &slot : slots_)
    Release(&slot);
  matting_.reset();
  matting_alpha_.Free();
  shutdown_ = true;
  return true;
}

bool QualifiedFinalResidentExecutor::
    RetireFailedFrameAfterQuiescence() noexcept {
  if (model_effects_) {
    auto &batch = model_effects_->batch;
    if (batch.state() == ResidentCommandBatchState::recording)
      batch.AbortUnsubmitted();
    if (batch.state() == ResidentCommandBatchState::unsafe_pending &&
        !batch.RetireAfterQuiescence(nullptr))
      return false;
  }
  if (kernels_) {
    kernels_->EndDeferredFrame();
    if (kernels_->deferred_frame_active())
      return false;
  }
  return true;
}

} // namespace studiocast::vulkan
