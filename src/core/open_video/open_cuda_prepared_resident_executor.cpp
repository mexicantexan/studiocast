#include "core/open_video/open_cuda_prepared_resident_executor.h"

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/kernels/open_cuda_vb_kernels.h"
#include "core/cuda/kernels/prepared_open_cuda_vb_kernel_bundle.h"
#include "core/cuda/kernels/prepared_preprocess_kernel_bundle.h"
#include "core/cuda/kernels/resize_bilinear.h"
#include "core/cuda/kernels/prepared_vignette.h"
#include "core/open_video/matting_session.h"
#include "core/video/effects/broadcast_effect_contract.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace studiocast::open_cuda {
namespace {

using hal::BackendId;
using hal::FailureAction;
using hal::ReasonId;
using hal::SemanticEffectId;
using hal::SemanticEffectProviderId;
using hal::StageExecutionStatus;
using hal::StatusId;
using RuntimeStatus = PreparedOpenCudaRuntimeStatus;

constexpr std::size_t kRuntimeSlots = 4;
constexpr std::uint32_t kKernelBlock = 16;
// The original strict resident island owns 16 bounded resources. Geometry
// parity always adds one resized-alpha image, optionally adds the separable
// feather pair, and blur owns two additional RGB scratch images.
constexpr std::uint32_t kBaseRuntimeResourceCount = 16;
constexpr std::uint32_t kPreparedResizedAlphaImageCount = 1;
constexpr std::uint32_t kPreparedFeatherImageCount = 2;
constexpr std::uint32_t kPreparedRgbBlurImageCount = 2;
constexpr std::uint32_t kPreparedReplacementImageCount = 2;
constexpr auto kOrtProvider =
    SemanticEffectProviderId::open_video_onnxruntime_cuda;
constexpr auto kNativeProvider =
    SemanticEffectProviderId::open_cuda_production_kernels;

constexpr std::uint32_t
PreparedResourceCount(bool blur, bool feather, bool replace,
                      bool final_resize = false) noexcept {
  return kBaseRuntimeResourceCount + kPreparedResizedAlphaImageCount +
         (feather ? kPreparedFeatherImageCount : 0) +
         (blur ? kPreparedRgbBlurImageCount : 0) +
         (replace ? kPreparedReplacementImageCount : 0) +
         (final_resize ? 1u : 0u);
}

bool SameHalModel(
    const hal::ModelArtifactIdentity &model,
    const open_video::ValidatedModelArtifactIdentity &artifact) noexcept {
  return artifact.Valid() && model.model_id_hash == artifact.model_id_sha256 &&
         model.artifact_set_hash == artifact.artifact_set_sha256 &&
         model.artifact_generation == artifact.registry_generation;
}

bool UsesMatting(SemanticEffectId semantic) noexcept {
  return semantic != SemanticEffectId::vignette;
}

SemanticEffectProviderId ProviderFor(SemanticEffectId semantic) noexcept {
  return UsesMatting(semantic) ? kOrtProvider : kNativeProvider;
}

bool ZeroModel(const hal::ModelArtifactIdentity &model) noexcept {
  return model.model_id_hash == std::array<std::uint8_t, 32>{} &&
         model.artifact_set_hash == std::array<std::uint8_t, 32>{} &&
         model.artifact_generation == 0;
}

bool SameResource(const hal::ResourceHandle &lhs,
                  const hal::ResourceHandle &rhs) noexcept {
  return lhs.resource_id == rhs.resource_id &&
         lhs.allocator_owner_id == rhs.allocator_owner_id &&
         lhs.lifetime_generation == rhs.lifetime_generation &&
         lhs.memory_domain == rhs.memory_domain &&
         lhs.lifetime == rhs.lifetime &&
         lhs.synchronization_state == rhs.synchronization_state &&
         lhs.execution_domain == rhs.execution_domain &&
         lhs.size_bytes == rhs.size_bytes &&
         lhs.alignment_bytes == rhs.alignment_bytes &&
         lhs.mutable_resource == rhs.mutable_resource &&
         lhs.aliasable == rhs.aliasable &&
         lhs.adapter_opaque == rhs.adapter_opaque;
}

bool SameRuntimeHandle(const PreparedOpenCudaRuntimeHandle &lhs,
                       const PreparedOpenCudaRuntimeHandle &rhs) noexcept {
  return lhs.handle_id == rhs.handle_id && lhs.program_id == rhs.program_id &&
         lhs.invalidation_key == rhs.invalidation_key &&
         lhs.execution_domain == rhs.execution_domain &&
         lhs.opaque == rhs.opaque;
}

bool BudgetContains(const hal::ResourceBudget &available,
                    const hal::ResourceBudget &required) noexcept {
  return available.persistent_bytes >= required.persistent_bytes &&
         available.scratch_bytes >= required.scratch_bytes &&
         available.upload_staging_bytes >= required.upload_staging_bytes &&
         available.readback_staging_bytes >= required.readback_staging_bytes &&
         available.resource_count >= required.resource_count &&
         available.in_flight_slots >= required.in_flight_slots;
}

bool SameBudget(const hal::ResourceBudget &lhs,
                const hal::ResourceBudget &rhs) noexcept {
  return lhs.persistent_bytes == rhs.persistent_bytes &&
         lhs.scratch_bytes == rhs.scratch_bytes &&
         lhs.upload_staging_bytes == rhs.upload_staging_bytes &&
         lhs.readback_staging_bytes == rhs.readback_staging_bytes &&
         lhs.resource_count == rhs.resource_count &&
         lhs.in_flight_slots == rhs.in_flight_slots;
}

bool CheckedMul(std::uint64_t lhs, std::uint64_t rhs,
                std::uint64_t *out) noexcept {
  if (!out ||
      (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

bool CheckedAdd(std::uint64_t lhs, std::uint64_t rhs,
                std::uint64_t *out) noexcept {
  if (!out || rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    return false;
  *out = lhs + rhs;
  return true;
}

PreparedOpenCudaVbPlan BuildPreparedOpenCudaVirtualBackgroundPlan(
    const PreparedOpenCudaVbPlanInput &input) noexcept {
  PreparedOpenCudaVbPlan result;
  result.semantic = input.semantic;
  const bool blur = input.semantic == SemanticEffectId::virtual_background_blur;
  const bool remove =
      input.semantic == SemanticEffectId::virtual_background_remove;
  const bool replace =
      input.semantic == SemanticEffectId::virtual_background_replace;
  const bool key_light = input.semantic == SemanticEffectId::virtual_key_light;
  const bool final_resize = input.final_output_width != 0 &&
                            (input.final_output_width != input.frame_width ||
                             input.final_output_height != input.frame_height);
  if ((!blur && !remove && !replace && !key_light) ||
      input.island_stage_count == 0 || input.island_stage_count > 2) {
    result.status = RuntimeStatus::unsupported_effect;
    result.reason = ReasonId::semantic_effect_unsupported;
    return result;
  }
  const int clamped_strength = std::clamp(
      input.requested_strength, video::effects::contract::kVbStrengthMin,
      video::effects::contract::kVbStrengthMax);
  const std::uint32_t feather_radius =
      static_cast<std::uint32_t>(std::min(4, clamped_strength / 16));
  const bool feather = !key_light && feather_radius != 0;
  if (input.frame_width == 0 || input.frame_height == 0 ||
      input.model_width == 0 || input.model_height == 0 ||
      (final_resize &&
       (input.final_output_height == 0 ||
        input.final_output_pitch_bytes < input.final_output_width * 3u)) ||
      input.frame_width >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      input.frame_height >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      input.model_width >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      input.model_height >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    result.status = RuntimeStatus::resource_exhausted;
    result.reason = ReasonId::resource_exhausted;
    return result;
  }

  std::uint64_t frame_rgb_row = 0;
  std::uint64_t frame_alpha_row = 0;
  std::uint64_t frame_rgb_active = 0;
  std::uint64_t model_alpha_active = 0;
  std::uint64_t input_tensor_active = 0;
  if (!CheckedMul(input.frame_width, 3, &frame_rgb_row) ||
      !CheckedMul(input.frame_width, sizeof(float), &frame_alpha_row) ||
      !CheckedMul(frame_rgb_row, input.frame_height, &frame_rgb_active) ||
      !CheckedMul(input.model_width, input.model_height, &model_alpha_active) ||
      !CheckedMul(model_alpha_active, sizeof(float), &model_alpha_active) ||
      !CheckedMul(model_alpha_active, 3, &input_tensor_active)) {
    result.status = RuntimeStatus::resource_exhausted;
    result.reason = ReasonId::resource_exhausted;
    return result;
  }
  if (input.input_pitch_bytes < frame_rgb_row ||
      input.output_pitch_bytes < frame_rgb_row ||
      input.model_alpha_capacity_bytes < model_alpha_active ||
      input.input_tensor_capacity_bytes < input_tensor_active ||
      input.resized_alpha_pitch_bytes < frame_alpha_row ||
      (feather && (input.alpha_scratch_pitch_bytes < frame_alpha_row ||
                   input.alpha_feather_pitch_bytes < frame_alpha_row)) ||
      input.upload_staging_bytes < frame_rgb_active ||
      input.readback_staging_bytes <
          (final_resize ? input.final_output_width *
                              input.final_output_height * 3u
                        : frame_rgb_active) ||
      (blur && (input.blur_scratch_pitch_bytes < frame_rgb_row ||
                input.blurred_pitch_bytes < frame_rgb_row)) ||
      (replace && (input.replacement_source_pitch_bytes == 0 ||
                   input.replacement_source_height == 0 ||
                   input.replacement_frame_pitch_bytes < frame_rgb_row))) {
    result.status = RuntimeStatus::malformed_contract;
    result.reason = ReasonId::malformed_descriptor;
    return result;
  }

  std::uint64_t input_capacity = 0;
  std::uint64_t output_capacity = 0;
  std::uint64_t resized_alpha_capacity = 0;
  std::uint64_t alpha_scratch_capacity = 0;
  std::uint64_t alpha_feather_capacity = 0;
  std::uint64_t blur_scratch_capacity = 0;
  std::uint64_t blurred_capacity = 0;
  std::uint64_t replacement_source_capacity = 0;
  std::uint64_t replacement_frame_capacity = 0;
  std::uint64_t final_output_capacity = 0;
  if (!CheckedMul(input.input_pitch_bytes, input.frame_height,
                  &input_capacity) ||
      !CheckedMul(input.output_pitch_bytes, input.frame_height,
                  &output_capacity) ||
      !CheckedMul(input.resized_alpha_pitch_bytes, input.frame_height,
                  &resized_alpha_capacity) ||
      (feather && (!CheckedMul(input.alpha_scratch_pitch_bytes,
                               input.frame_height, &alpha_scratch_capacity) ||
                   !CheckedMul(input.alpha_feather_pitch_bytes,
                               input.frame_height, &alpha_feather_capacity))) ||
      (blur && (!CheckedMul(input.blur_scratch_pitch_bytes, input.frame_height,
                            &blur_scratch_capacity) ||
                !CheckedMul(input.blurred_pitch_bytes, input.frame_height,
                            &blurred_capacity))) ||
      (replace &&
       (!CheckedMul(input.replacement_source_pitch_bytes,
                    input.replacement_source_height,
                    &replacement_source_capacity) ||
        !CheckedMul(input.replacement_frame_pitch_bytes, input.frame_height,
                    &replacement_frame_capacity))) ||
      (final_resize &&
       !CheckedMul(input.final_output_pitch_bytes, input.final_output_height,
                   &final_output_capacity))) {
    result.status = RuntimeStatus::resource_exhausted;
    result.reason = ReasonId::resource_exhausted;
    return result;
  }

  auto add = [](std::uint64_t value, std::uint64_t *total) noexcept {
    return CheckedAdd(*total, value, total);
  };
  auto &required = result.required_budget;
  if (!add(input_capacity, &required.persistent_bytes) ||
      !add(output_capacity, &required.persistent_bytes) ||
      !add(input.model_alpha_capacity_bytes, &required.persistent_bytes) ||
      !add(input.input_tensor_capacity_bytes, &required.persistent_bytes) ||
      !add(resized_alpha_capacity, &required.persistent_bytes) ||
      (feather && (!add(alpha_scratch_capacity, &required.scratch_bytes) ||
                   !add(alpha_feather_capacity, &required.scratch_bytes))) ||
      (blur && (!add(blur_scratch_capacity, &required.scratch_bytes) ||
                !add(blurred_capacity, &required.scratch_bytes))) ||
      (replace &&
       (!add(replacement_source_capacity, &required.persistent_bytes) ||
        !add(replacement_frame_capacity, &required.persistent_bytes))) ||
      (final_resize &&
       !add(final_output_capacity, &required.persistent_bytes))) {
    result.status = RuntimeStatus::resource_exhausted;
    result.reason = ReasonId::resource_exhausted;
    return result;
  }
  required.upload_staging_bytes = input.upload_staging_bytes;
  required.readback_staging_bytes = input.readback_staging_bytes;
  required.resource_count =
      PreparedResourceCount(blur, feather, replace, final_resize) +
      2u * (input.island_stage_count - 1u);
  required.in_flight_slots = 1;
  if (!BudgetContains(input.available_budget, required)) {
    result.status = RuntimeStatus::resource_exhausted;
    result.reason = ReasonId::resource_exhausted;
    return result;
  }

  result.frame_width = static_cast<std::uint32_t>(input.frame_width);
  result.island_stage_count = input.island_stage_count;
  result.frame_height = static_cast<std::uint32_t>(input.frame_height);
  result.final_output_width = static_cast<std::uint32_t>(
      input.final_output_width == 0 ? input.frame_width
                                    : input.final_output_width);
  result.final_output_height = static_cast<std::uint32_t>(
      input.final_output_height == 0 ? input.frame_height
                                     : input.final_output_height);
  result.model_width = static_cast<std::uint32_t>(input.model_width);
  result.model_height = static_cast<std::uint32_t>(input.model_height);
  result.strength = static_cast<std::uint32_t>(clamped_strength);
  result.alpha_feather_radius = feather ? feather_radius : 0;

  const std::uint32_t grid_x =
      result.frame_width / kKernelBlock +
      (result.frame_width % kKernelBlock != 0 ? 1u : 0u);
  const std::uint32_t grid_y =
      result.frame_height / kKernelBlock +
      (result.frame_height % kKernelBlock != 0 ? 1u : 0u);
  const auto append = [&](PreparedOpenCudaOperationId operation) noexcept {
    auto &launch = result.launches[result.launch_count++];
    launch.operation = operation;
    launch.grid_x = grid_x;
    launch.grid_y = grid_y;
    launch.block_x = kKernelBlock;
    launch.block_y = kKernelBlock;
  };
  append(PreparedOpenCudaOperationId::alpha_resize);
  if (result.alpha_feather_radius != 0) {
    append(PreparedOpenCudaOperationId::alpha_feather_horizontal);
    append(PreparedOpenCudaOperationId::alpha_feather_vertical);
  }
  if (blur) {
    append(PreparedOpenCudaOperationId::background_blur_horizontal);
    append(PreparedOpenCudaOperationId::background_blur_vertical);
    append(PreparedOpenCudaOperationId::composite_background);
  } else if (remove) {
    append(PreparedOpenCudaOperationId::composite_solid);
  } else if (replace) {
    append(PreparedOpenCudaOperationId::composite_replace);
  } else {
    append(PreparedOpenCudaOperationId::key_light);
  }
  result.status = RuntimeStatus::ready;
  result.reason = ReasonId::none;
  return result;
}

bool ExactPreparedOpenCudaVirtualBackgroundPlan(
    const PreparedOpenCudaVbPlan &plan) noexcept {
  const bool blur = plan.semantic == SemanticEffectId::virtual_background_blur;
  const bool remove =
      plan.semantic == SemanticEffectId::virtual_background_remove;
  const bool replace =
      plan.semantic == SemanticEffectId::virtual_background_replace;
  const bool key_light = plan.semantic == SemanticEffectId::virtual_key_light;
  const bool feather = plan.alpha_feather_radius != 0;
  const bool final_resize = plan.final_output_width != plan.frame_width ||
                            plan.final_output_height != plan.frame_height;
  if (!plan.ready() || plan.frame_width == 0 || plan.frame_height == 0 ||
      plan.model_width == 0 || plan.model_height == 0 ||
      plan.island_stage_count == 0 || plan.island_stage_count > 2 ||
      plan.required_budget.persistent_bytes == 0 ||
      ((blur || feather) && plan.required_budget.scratch_bytes == 0) ||
      plan.required_budget.upload_staging_bytes == 0 ||
      plan.required_budget.readback_staging_bytes == 0 ||
      plan.required_budget.in_flight_slots != 1 ||
      plan.strength < static_cast<std::uint32_t>(
                          video::effects::contract::kVbStrengthMin) ||
      plan.strength > static_cast<std::uint32_t>(
                          video::effects::contract::kVbStrengthMax) ||
      plan.alpha_feather_radius !=
          (key_light ? 0u : std::min(4u, plan.strength / 16u)) ||
      (!blur && !remove && !replace && !key_light) ||
      plan.required_budget.resource_count !=
          PreparedResourceCount(blur, feather, replace, final_resize) +
              2u * (plan.island_stage_count - 1u)) {
    return false;
  }
  std::array<PreparedOpenCudaOperationId, PreparedOpenCudaVbPlan::kMaxLaunches>
      expected{};
  std::uint32_t expected_count = 0;
  expected[expected_count++] = PreparedOpenCudaOperationId::alpha_resize;
  if (plan.alpha_feather_radius != 0) {
    expected[expected_count++] =
        PreparedOpenCudaOperationId::alpha_feather_horizontal;
    expected[expected_count++] =
        PreparedOpenCudaOperationId::alpha_feather_vertical;
  }
  if (blur) {
    expected[expected_count++] =
        PreparedOpenCudaOperationId::background_blur_horizontal;
    expected[expected_count++] =
        PreparedOpenCudaOperationId::background_blur_vertical;
    expected[expected_count++] =
        PreparedOpenCudaOperationId::composite_background;
  } else if (remove) {
    expected[expected_count++] = PreparedOpenCudaOperationId::composite_solid;
  } else if (replace) {
    expected[expected_count++] = PreparedOpenCudaOperationId::composite_replace;
  } else if (key_light) {
    expected[expected_count++] = PreparedOpenCudaOperationId::key_light;
  } else {
    return false;
  }
  if (plan.launch_count != expected_count)
    return false;
  const std::uint32_t expected_grid_x =
      plan.frame_width / kKernelBlock +
      (plan.frame_width % kKernelBlock != 0 ? 1u : 0u);
  const std::uint32_t expected_grid_y =
      plan.frame_height / kKernelBlock +
      (plan.frame_height % kKernelBlock != 0 ? 1u : 0u);
  for (std::uint32_t index = 0; index < plan.launch_count; ++index) {
    const auto &launch = plan.launches[index];
    if (launch.operation != expected[index] ||
        launch.grid_x != expected_grid_x || launch.grid_y != expected_grid_y ||
        launch.block_x != kKernelBlock || launch.block_y != kKernelBlock) {
      return false;
    }
  }
  return true;
}

bool TightRgb(const hal::ImageDescriptor &image, hal::HostCoherence coherence,
              hal::Mutability mutability) noexcept {
  if (!hal::ValidateImageDescriptor(image) || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != coherence || image.mutability != mutability ||
      image.aliasing != hal::AliasingPermission::forbidden ||
      image.in_place_permitted) {
    return false;
  }
  std::uint64_t row_bytes = 0;
  std::uint64_t size_bytes = 0;
  return CheckedMul(image.width, 3, &row_bytes) &&
         CheckedMul(row_bytes, image.height, &size_bytes) &&
         image.planes[0].offset_bytes == 0 &&
         image.planes[0].width == image.width &&
         image.planes[0].height == image.height &&
         image.planes[0].row_stride_bytes == row_bytes &&
         image.planes[0].size_bytes == size_bytes;
}

bool ResidentRgb(const hal::ImageDescriptor &image,
                 hal::Mutability mutability) noexcept {
  if (!hal::ValidateImageDescriptor(image) || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != hal::HostCoherence::not_host_visible ||
      image.mutability != mutability ||
      image.aliasing != hal::AliasingPermission::forbidden ||
      image.in_place_permitted || image.planes[0].offset_bytes != 0 ||
      image.planes[0].width != image.width ||
      image.planes[0].height != image.height) {
    return false;
  }
  std::uint64_t minimum_row = 0;
  std::uint64_t expected_size = 0;
  return CheckedMul(image.width, 3, &minimum_row) &&
         image.planes[0].row_stride_bytes >= minimum_row &&
         CheckedMul(image.planes[0].row_stride_bytes, image.height,
                    &expected_size) &&
         image.planes[0].size_bytes == expected_size;
}

bool ExactCudaDomain(const hal::ExecutionDomainId &domain,
                     std::uint64_t adapter_id,
                     std::uint64_t stream_identity) noexcept {
  return hal::ValidateExecutionDomain(domain) &&
         domain.backend == BackendId::cuda &&
         domain.api_family == hal::ApiFamily::cuda_driver &&
         domain.allocator_family == hal::AllocatorFamily::cuda_driver &&
         domain.memory_handle_family ==
             hal::MemoryHandleFamily::cuda_device_pointer &&
         domain.ownership_model == hal::OwnershipModel::adapter_exclusive &&
         domain.synchronization_protocol ==
             hal::SynchronizationProtocol::cuda_stream_ordered &&
         domain.adapter_instance_id == adapter_id &&
         domain.queue_or_stream_identity == stream_identity;
}

ReasonId ValidateProductionProgramInternal(
    const hal::PreparedProgram &program,
    const PreparedOpenCudaProductionConfig &config) noexcept {
  if (program.stage_count == 0 || program.stage_count > 3 ||
      program.cpu_tail_count != 1 || program.temporal.temporal ||
      program.temporal.history_frames != 0 || !program.requires_upload ||
      !program.requires_final_readback ||
      program.requested_backend != BackendId::cuda ||
      !program.strict_execution_domain ||
      program.requested_execution_domain != program.execution_domain) {
    return ReasonId::malformed_contract;
  }
  if (!ExactCudaDomain(program.execution_domain, config.adapter_instance_id,
                       config.stream_identity) ||
      program.owner_adapter_instance_id != config.adapter_instance_id ||
      !config.prepared_stream || !config.executor_owns_stream ||
      config.stream_identity !=
          reinterpret_cast<std::uintptr_t>(config.prepared_stream)) {
    return ReasonId::wrong_execution_domain;
  }
  const auto vb = [](SemanticEffectId semantic) {
    return semantic == SemanticEffectId::virtual_background_blur ||
           semantic == SemanticEffectId::virtual_background_remove ||
           semantic == SemanticEffectId::virtual_background_replace;
  };
  const bool vignette_last = program.stages[program.stage_count - 1].semantic ==
                             SemanticEffectId::vignette;
  const std::size_t model_count = program.stage_count - (vignette_last ? 1u : 0u);
  if (!((model_count == 0 && vignette_last) ||
        (model_count == 1 &&
         (vb(program.stages[0].semantic) ||
          program.stages[0].semantic == SemanticEffectId::virtual_key_light)) ||
        (model_count == 2 && vb(program.stages[0].semantic) &&
         program.stages[1].semantic == SemanticEffectId::virtual_key_light))) {
    return ReasonId::semantic_effect_unsupported;
  }
  const auto aggregate_provider =
      model_count != 0 && vignette_last ? SemanticEffectProviderId::invalid
      : model_count != 0 ? kOrtProvider : kNativeProvider;
  if (program.requested_provider != aggregate_provider)
    return ReasonId::semantic_provider_unavailable;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.backend != BackendId::cuda ||
        stage.provider != ProviderFor(stage.semantic) ||
        stage.execution_domain != program.execution_domain)
      return ReasonId::semantic_provider_unavailable;
    if ((stage.required &&
         stage.failure_action != FailureAction::fail_closed) ||
        (!stage.required &&
         stage.failure_action != FailureAction::fail_open_preserve_input))
      return ReasonId::malformed_contract;
    if ((UsesMatting(stage.semantic) &&
         !SameHalModel(stage.model, config.model_artifact)) ||
        (!UsesMatting(stage.semantic) && !ZeroModel(stage.model)))
      return ReasonId::reconfiguration_required;
  }
  if ((model_count != 0 &&
       (program.invalidation_key.model_artifact_hash == 0 ||
        program.invalidation_key.model_artifact_hash != config.model_artifact_hash)) ||
      (model_count == 0 &&
       (program.invalidation_key.model_artifact_hash != 0 ||
        config.model_artifact_hash != 0)) ||
      program.invalidation_key.generations.configuration !=
          config.configuration_generation ||
      (model_count != 0 &&
       program.invalidation_key.generations.model !=
           config.model_preprocess_generation) ||
      program.invalidation_key.generations.context !=
          program.execution_domain.logical_context_generation) {
    return ReasonId::reconfiguration_required;
  }
  if (!TightRgb(program.host_input, hal::HostCoherence::coherent,
                hal::Mutability::read_only) ||
      !TightRgb(program.host_output, hal::HostCoherence::coherent,
                hal::Mutability::write_only) ||
      program.host_input.width == 0 || program.host_output.width == 0) {
    return ReasonId::malformed_descriptor;
  }
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    const auto &expected_output =
        index + 1 == program.stage_count ? program.host_output
                                         : program.host_input;
    if (stage.input.kind != hal::StageIoKind::image ||
        stage.output.kind != hal::StageIoKind::image ||
        !ResidentRgb(stage.input.image, hal::Mutability::read_only) ||
        !ResidentRgb(stage.output.image, hal::Mutability::read_write) ||
        stage.input.image.width != program.host_input.width ||
        stage.input.image.height != program.host_input.height ||
        stage.output.image.width != expected_output.width ||
        stage.output.image.height != expected_output.height)
      return ReasonId::malformed_descriptor;
  }
  const auto &stage = program.stages[program.stage_count - 1];
  const auto &tail = program.cpu_tails[0];
  if (tail.id != hal::CpuTailId::writer_copy ||
      tail.after_stage_id != stage.stage_id || !tail.required ||
      tail.placement != hal::DeclaredCpuTail::Placement::after_stage ||
      (tail.boundary_index != 0 &&
       tail.boundary_index != program.stage_count)) {
    return ReasonId::cpu_tail_required;
  }
  if (model_count == 0)
    return (!config.model_artifact.Valid() && config.model_preprocess_generation == 0)
               ? ReasonId::none
               : ReasonId::capability_evidence_incomplete;
  if (!config.model_pack || !config.model_artifact.Valid() ||
      config.model_pack->registry_generation !=
          config.model_artifact.registry_generation ||
      config.model_pack->registry_manifest_sha256 !=
          config.model_artifact.manifest_sha256 ||
      config.model_pack->task != "matting" ||
      !config.model_pack->matting.has_value()) {
    return ReasonId::capability_evidence_incomplete;
  }
  const auto &matting = *config.model_pack->matting;
  if (matting.input.width <= 0 || matting.input.height <= 0 ||
      matting.input.channels != 3 || matting.input.layout != "nchw" ||
      matting.input.dtype != "float32" || matting.output.kind != "alpha" ||
      matting.output.dtype != "float32" ||
      (matting.output.width != 0 &&
       matting.output.width != matting.input.width) ||
      (matting.output.height != 0 &&
       matting.output.height != matting.input.height) ||
      (matting.output.channels != 0 && matting.output.channels != 1) ||
      matting.preprocess.color != "rgb" || matting.preprocess.range != "0..1") {
    return ReasonId::capability_evidence_incomplete;
  }
  return ReasonId::none;
}

StatusId HalStatus(RuntimeStatus status) noexcept {
  switch (status) {
  case RuntimeStatus::ready:
    return StatusId::ready;
  case RuntimeStatus::executed:
    return StatusId::executed;
  case RuntimeStatus::optional_failed_open:
    return StatusId::degraded;
  case RuntimeStatus::resource_exhausted:
  case RuntimeStatus::completion_pending:
    return StatusId::resource_exhausted;
  case RuntimeStatus::stale_generation:
  case RuntimeStatus::stale_artifact:
  case RuntimeStatus::foreign_handle:
    return StatusId::stale;
  case RuntimeStatus::device_lost:
  case RuntimeStatus::quarantined:
    return StatusId::quarantined;
  case RuntimeStatus::stop_requested:
  case RuntimeStatus::cancelled:
  case RuntimeStatus::shutdown:
    return StatusId::stopped;
  case RuntimeStatus::invalid:
  case RuntimeStatus::required_failed:
  case RuntimeStatus::malformed_contract:
  case RuntimeStatus::unsupported_effect:
  case RuntimeStatus::wrong_provider:
  case RuntimeStatus::wrong_execution_domain:
  case RuntimeStatus::not_prepared:
  case RuntimeStatus::host_binding_invalid:
  case RuntimeStatus::preprocess_failed:
  case RuntimeStatus::inference_failed:
  case RuntimeStatus::postprocess_failed:
  case RuntimeStatus::transfer_failed:
  case RuntimeStatus::synchronization_failed:
  case RuntimeStatus::timeout:
    return StatusId::failed;
  }
  return StatusId::failed;
}

ReasonId HalReason(RuntimeStatus status) noexcept {
  switch (status) {
  case RuntimeStatus::ready:
  case RuntimeStatus::executed:
    return ReasonId::none;
  case RuntimeStatus::optional_failed_open:
    return ReasonId::optional_stage_failed_open;
  case RuntimeStatus::required_failed:
    return ReasonId::required_stage_failed;
  case RuntimeStatus::malformed_contract:
    return ReasonId::malformed_contract;
  case RuntimeStatus::unsupported_effect:
    return ReasonId::semantic_effect_unsupported;
  case RuntimeStatus::wrong_provider:
    return ReasonId::semantic_provider_unavailable;
  case RuntimeStatus::wrong_execution_domain:
    return ReasonId::wrong_execution_domain;
  case RuntimeStatus::stale_generation:
  case RuntimeStatus::stale_artifact:
    return ReasonId::reconfiguration_required;
  case RuntimeStatus::resource_exhausted:
  case RuntimeStatus::completion_pending:
    return ReasonId::resource_exhausted;
  case RuntimeStatus::not_prepared:
  case RuntimeStatus::foreign_handle:
    return ReasonId::stale_program;
  case RuntimeStatus::host_binding_invalid:
    return ReasonId::host_frame_binding_invalid;
  case RuntimeStatus::preprocess_failed:
  case RuntimeStatus::postprocess_failed:
  case RuntimeStatus::transfer_failed:
    return ReasonId::transfer_contract_violated;
  case RuntimeStatus::inference_failed:
    return ReasonId::semantic_provider_unavailable;
  case RuntimeStatus::device_lost:
  case RuntimeStatus::quarantined:
    return ReasonId::device_lost;
  case RuntimeStatus::stop_requested:
  case RuntimeStatus::cancelled:
  case RuntimeStatus::shutdown:
    return ReasonId::shutdown_requested;
  case RuntimeStatus::invalid:
  case RuntimeStatus::synchronization_failed:
  case RuntimeStatus::timeout:
    return ReasonId::synchronization_failed;
  }
  return ReasonId::synchronization_failed;
}

std::uint64_t MonotonicNowNanoseconds(
    const PreparedOpenCudaProductionConfig &config) noexcept {
  if (config.wait_policy.now_nanoseconds) {
    return config.wait_policy.now_nanoseconds(
        config.wait_policy.callback_context);
  }
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void FinishFrameTiming(const PreparedOpenCudaProductionConfig &config,
                       std::uint64_t started_ns,
                       PreparedOpenCudaRuntimeFrameResult *result) noexcept {
  if (!result)
    return;
  const std::uint64_t finished_ns = MonotonicNowNanoseconds(config);
  const std::uint64_t elapsed =
      finished_ns >= started_ns ? finished_ns - started_ns : 0;
  result->frame_elapsed_ns = std::max(result->stage_elapsed_ns, elapsed);
}

RuntimeStatus
TransferStatus(cuda::CudaPreparedTransferCompletion::Status status) noexcept {
  using Transfer = cuda::CudaPreparedTransferCompletion;
  switch (status) {
  case Transfer::Status::ok:
    return RuntimeStatus::executed;
  case Transfer::Status::resource_exhausted:
  case Transfer::Status::already_prepared:
    return RuntimeStatus::resource_exhausted;
  case Transfer::Status::device_lost:
    return RuntimeStatus::device_lost;
  case Transfer::Status::timeout:
    return RuntimeStatus::timeout;
  case Transfer::Status::stop_requested:
    return RuntimeStatus::stop_requested;
  case Transfer::Status::cancelled:
    return RuntimeStatus::cancelled;
  case Transfer::Status::shutdown:
    return RuntimeStatus::shutdown;
  case Transfer::Status::invalid_context:
  case Transfer::Status::stale_context:
  case Transfer::Status::foreign_context:
  case Transfer::Status::foreign_owner:
  case Transfer::Status::wrong_stream:
    return RuntimeStatus::stale_generation;
  case Transfer::Status::invalid_argument:
  case Transfer::Status::invalid_sequence:
    return RuntimeStatus::malformed_contract;
  case Transfer::Status::unsupported:
  case Transfer::Status::not_prepared:
  case Transfer::Status::copy_failed:
  case Transfer::Status::event_failed:
  case Transfer::Status::resource_release_failed:
    return RuntimeStatus::transfer_failed;
  }
  return RuntimeStatus::transfer_failed;
}

bool ForbiddenTelemetryUnchanged(
    const PreparedOpenCudaRuntimeTelemetry &before,
    const PreparedOpenCudaRuntimeTelemetry &after) noexcept {
  return before.allocation_requests == after.allocation_requests &&
         before.provider_probes == after.provider_probes &&
         before.model_scans == after.model_scans &&
         before.manifest_parses == after.manifest_parses &&
         before.session_constructions == after.session_constructions &&
         before.json_serializations == after.json_serializations;
}

hal::ResourceHandle
MakeResource(const hal::PreparedProgram &program, std::uint64_t resource_id,
             hal::MemoryDomain domain, hal::ResourceLifetime lifetime,
             std::uint64_t size_bytes, std::uint32_t alignment,
             std::uint64_t opaque) noexcept {
  hal::ResourceHandle resource;
  resource.resource_id = resource_id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = domain;
  resource.lifetime = lifetime;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = size_bytes;
  resource.alignment_bytes = alignment;
  resource.mutable_resource = true;
  resource.aliasable = false;
  resource.adapter_opaque[0] = opaque;
  return resource;
}

bool SamePhysicalDevice(
    const hal::PhysicalDeviceIdentity &domain,
    const cuda::CudaPhysicalDeviceIdentity &cuda_identity) noexcept {
  return domain.stable_id == cuda_identity.uuid;
}

cuda::kernels::ModelPreprocessSpec
PreprocessSpec(const open_video::ModelPack &pack) noexcept {
  cuda::kernels::ModelPreprocessSpec result;
  if (!pack.matting)
    return result;
  const auto &source = *pack.matting;
  result.dst_w = source.input.width;
  result.dst_h = source.input.height;
  for (std::size_t index = 0; index < 3; ++index) {
    result.mean[index] = static_cast<float>(source.preprocess.mean[index]);
    result.std[index] = static_cast<float>(source.preprocess.std[index]);
  }
  result.dst_order = cuda::kernels::ChannelOrder::rgb;
  return result;
}

class ProductionPreparedOpenCudaRuntime final
    : public IPreparedOpenCudaRuntime {
public:
  ProductionPreparedOpenCudaRuntime(cuda::CudaDriverApi *driver,
                                    cuda::CUstream stream,
                                    bool owns_stream) noexcept
      : owned_driver_(driver), owned_stream_(stream), owns_stream_(owns_stream),
        stream_was_owned_(owns_stream) {
    shutdown_snapshot_.stream_owned = stream_was_owned_;
  }

  ~ProductionPreparedOpenCudaRuntime() override {
    (void)RequestShutdown();
    // If the driver still reports pending DMA, retaining the complete slot is
    // safer than destroying ORT/device/pinned storage underneath it. Context
    // teardown reclaims those process resources.
    for (auto &slot : slots_) {
      if (slot)
        (void)slot.release();
    }
  }

  PreparedOpenCudaRuntimePrepareResult
  Prepare(const hal::PreparedProgram &program,
          const PreparedOpenCudaProductionConfig &config) noexcept override {
    PreparedOpenCudaRuntimePrepareResult result;
    result.reason = ValidatePreparedOpenCudaProductionProgram(program, config);
    if (result.reason != ReasonId::none) {
      result.status = RuntimeStatus::malformed_contract;
      if (result.reason == ReasonId::semantic_effect_unsupported)
        result.status = RuntimeStatus::unsupported_effect;
      else if (result.reason == ReasonId::semantic_provider_unavailable)
        result.status = RuntimeStatus::wrong_provider;
      else if (result.reason == ReasonId::wrong_execution_domain)
        result.status = RuntimeStatus::wrong_execution_domain;
      else if (result.reason == ReasonId::reconfiguration_required)
        result.status = RuntimeStatus::stale_generation;
      return result;
    }
    if (shutdown_) {
      result.status = RuntimeStatus::shutdown;
      result.reason = ReasonId::shutdown_requested;
      return result;
    }
    auto destination = std::find_if(slots_.begin(), slots_.end(),
                                    [](const auto &entry) { return !entry; });
    if (destination == slots_.end()) {
      result.status = RuntimeStatus::resource_exhausted;
      result.reason = ReasonId::resource_exhausted;
      return result;
    }

    try {
      auto candidate = std::make_unique<Slot>();
      ++telemetry_.allocation_requests;
      candidate->program = program;
      candidate->config = config;
      candidate->config.model_pack = nullptr;
      const bool model_free = program.stages[0].semantic == SemanticEffectId::vignette;
      if (!model_free) {
        candidate->model_pack = *config.model_pack;
        ++telemetry_.allocation_requests;
        candidate->config.model_pack = &candidate->model_pack;
      }

      std::string error;
      if (!config.driver || !config.driver->IsInitialized() ||
          !config.driver->HasPreparedTransferCompletionSupport() ||
          !config.driver->EnsureContextForDevice(config.cuda_device_id,
                                                 &error)) {
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::semantic_provider_unavailable);
      }
      const auto *context = config.driver->context_identity();
      if (!context || !context->Valid() ||
          context->logical_generation !=
              program.execution_domain.logical_context_generation ||
          !SamePhysicalDevice(program.execution_domain.physical_device,
                              context->physical_device)) {
        return SetupFailure(RuntimeStatus::stale_generation,
                            ReasonId::wrong_generation);
      }
      candidate->context = *context;
      if (config.driver != owned_driver_ ||
          config.prepared_stream != owned_stream_ ||
          !config.executor_owns_stream || !owns_stream_)
        return SetupFailure(RuntimeStatus::wrong_execution_domain,
                            ReasonId::wrong_execution_domain);
      candidate->stream = config.prepared_stream;

      if (model_free) {
        if (config.driver->BindPreparedContext(candidate->context) !=
            cuda::CUDA_SUCCESS) {
          return SetupFailure(RuntimeStatus::wrong_execution_domain,
                              ReasonId::wrong_execution_domain);
        }
        cuda::CUcontext native_context = nullptr;
        if (!config.driver->f().cuCtxGetCurrent ||
            config.driver->f().cuCtxGetCurrent(&native_context) !=
                cuda::CUDA_SUCCESS ||
            !native_context) {
          return SetupFailure(RuntimeStatus::wrong_execution_domain,
                              ReasonId::wrong_execution_domain);
        }
        const int width = static_cast<int>(program.host_input.width);
        const int height = static_cast<int>(program.host_input.height);
        if (!candidate->input.Allocate(config.driver, width, height,
                                       cuda::PixelFormatGpu::rgb_u8, &error) ||
            !candidate->output.Allocate(config.driver, width, height,
                                        cuda::PixelFormatGpu::rgb_u8, &error) ||
            !cuda::kernels::PrepareVignetteKernel(
                *config.driver, native_context, candidate->stream,
                &candidate->vignette, &error)) {
          CleanupCandidate(candidate.get());
          return SetupFailure(RuntimeStatus::not_prepared,
                              ReasonId::capability_evidence_incomplete);
        }
        const bool final_resize = program.host_output.width != program.host_input.width ||
                                  program.host_output.height != program.host_input.height;
        if (final_resize &&
            (!cuda::kernels::PrepareResizeBilinearKernel(
                 *config.driver, native_context, candidate->stream,
                 &candidate->resize, &error) ||
             !candidate->final_output.Allocate(
                 config.driver, static_cast<int>(program.host_output.width),
                 static_cast<int>(program.host_output.height),
                 cuda::PixelFormatGpu::rgb_u8, &error))) {
          CleanupCandidate(candidate.get());
          return SetupFailure(RuntimeStatus::not_prepared,
                              ReasonId::capability_evidence_incomplete);
        }
        cuda::CudaPreparedTransferCompletion::PrepareOptions transfer_options;
        transfer_options.binding.context = *context;
        transfer_options.binding.stream = candidate->stream;
        transfer_options.binding.stream_identity = config.stream_identity;
        transfer_options.binding.owner_identity = config.adapter_instance_id;
        transfer_options.binding.configuration_generation = config.configuration_generation;
        transfer_options.upload_row_bytes = program.host_input.planes[0].row_stride_bytes;
        transfer_options.upload_rows = program.host_input.height;
        transfer_options.readback_row_bytes = program.host_output.planes[0].row_stride_bytes;
        transfer_options.readback_rows = program.host_output.height;
        transfer_options.timed_stage_count = static_cast<std::uint32_t>(program.stage_count);
        transfer_options.budget.max_pinned_host_bytes =
            program.host_input.planes[0].size_bytes + program.host_output.planes[0].size_bytes;
        transfer_options.budget.max_events =
            static_cast<std::uint32_t>(program.stage_count * 2u + 1u);
        const auto prepared = candidate->transfer.Prepare(config.driver, transfer_options);
        if (!prepared.ok()) {
          CleanupCandidate(candidate.get());
          return SetupFailure(TransferStatus(prepared.status),
                              HalReason(TransferStatus(prepared.status)));
        }
        candidate->transfer_binding = transfer_options.binding;
        candidate->reserved_budget = program.budget;
        candidate->handle.handle_id = next_handle_id_++;
        candidate->handle.program_id = program.program_id;
        candidate->handle.invalidation_key = program.invalidation_key;
        candidate->handle.execution_domain = program.execution_domain;
        candidate->handle.opaque[0] = reinterpret_cast<std::uintptr_t>(candidate->stream);
        result.status = RuntimeStatus::ready;
        result.reason = ReasonId::none;
        result.handle = candidate->handle;
        result.evidence.status = RuntimeStatus::ready;
        result.evidence.semantic = SemanticEffectId::vignette;
        result.evidence.provider = kNativeProvider;
        result.evidence.execution_domain = program.execution_domain;
        result.evidence.reserved_budget = program.budget;
        result.evidence.strict_cuda_only = true;
        result.evidence.user_stream_bound = true;
        result.evidence.cpu_provider_fallback_disabled = true;
        result.evidence.reusable_resources_prepared = true;
        result.evidence.setup_warmup_complete = true;
        result.evidence.device_postprocess = true;
        result.evidence.one_upload_one_readback = true;
        result.evidence.no_intermediate_crossings = true;
        result.evidence.single_in_flight = true;
        *destination = std::move(candidate);
        return result;
      }

      const auto &image = program.stages[0].input.image;
      const auto &matting = *candidate->model_pack.matting;
      const std::uint64_t frame_width = image.width;
      const std::uint64_t frame_height = image.height;
      const std::uint64_t model_width =
          static_cast<std::uint64_t>(matting.input.width);
      const std::uint64_t model_height =
          static_cast<std::uint64_t>(matting.input.height);
      std::uint64_t frame_alpha_row = 0;
      std::uint64_t model_alpha_elements = 0;
      std::uint64_t model_alpha_bytes = 0;
      std::uint64_t input_tensor_bytes = 0;
      if (!CheckedMul(frame_width, sizeof(float), &frame_alpha_row) ||
          !CheckedMul(model_width, model_height, &model_alpha_elements) ||
          !CheckedMul(model_alpha_elements, sizeof(float),
                      &model_alpha_bytes) ||
          !CheckedMul(model_alpha_bytes, 3, &input_tensor_bytes)) {
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      PreparedOpenCudaVbPlanInput preflight;
      preflight.island_stage_count =
          static_cast<std::uint32_t>(program.stage_count);
      preflight.semantic = program.stages[0].semantic;
      preflight.frame_width = frame_width;
      preflight.frame_height = frame_height;
      preflight.model_width = model_width;
      preflight.model_height = model_height;
      preflight.requested_strength = config.strength;
      preflight.input_pitch_bytes = image.planes[0].row_stride_bytes;
      preflight.output_pitch_bytes =
          program.stages[0].input.image.planes[0].row_stride_bytes;
      preflight.final_output_width = program.host_output.width;
      preflight.final_output_height = program.host_output.height;
      preflight.final_output_pitch_bytes =
          static_cast<std::uint64_t>(program.host_output.width) * 3u;
      preflight.model_alpha_capacity_bytes = model_alpha_bytes;
      preflight.input_tensor_capacity_bytes = input_tensor_bytes;
      preflight.resized_alpha_pitch_bytes = frame_alpha_row;
      const int clamped_strength =
          std::clamp(config.strength, video::effects::contract::kVbStrengthMin,
                     video::effects::contract::kVbStrengthMax);
      const bool replace = program.stages[0].semantic ==
                           SemanticEffectId::virtual_background_replace;
      const bool key_light =
          program.stages[0].semantic == SemanticEffectId::virtual_key_light;
      const bool feather =
          !key_light && std::min(4, clamped_strength / 16) != 0;
      preflight.alpha_scratch_pitch_bytes = feather ? frame_alpha_row : 0;
      preflight.alpha_feather_pitch_bytes = feather ? frame_alpha_row : 0;
      preflight.blur_scratch_pitch_bytes = image.planes[0].row_stride_bytes;
      preflight.blurred_pitch_bytes = image.planes[0].row_stride_bytes;
      if (replace) {
        std::uint64_t replacement_bytes = 0;
        if (config.replacement_width <= 0 || config.replacement_height <= 0 ||
            !CheckedMul(static_cast<std::uint64_t>(config.replacement_width),
                        3u, &preflight.replacement_source_pitch_bytes) ||
            !CheckedMul(preflight.replacement_source_pitch_bytes,
                        static_cast<std::uint64_t>(config.replacement_height),
                        &replacement_bytes) ||
            replacement_bytes != config.replacement_rgb.size()) {
          return SetupFailure(RuntimeStatus::malformed_contract,
                              ReasonId::malformed_descriptor);
        }
        preflight.replacement_source_height =
            static_cast<std::uint64_t>(config.replacement_height);
        preflight.replacement_frame_pitch_bytes =
            image.planes[0].row_stride_bytes;
      }
      preflight.upload_staging_bytes = program.host_input.planes[0].size_bytes;
      preflight.readback_staging_bytes =
          program.host_output.planes[0].size_bytes;
      preflight.available_budget = program.budget;
      const auto minimum_plan =
          BuildPreparedOpenCudaVirtualBackgroundPlan(preflight);
      if (!minimum_plan.ready())
        return SetupFailure(minimum_plan.status, minimum_plan.reason);

      const int width = static_cast<int>(image.width);
      const int height = static_cast<int>(image.height);
      const int model_w = matting.input.width;
      const int model_h = matting.input.height;
      if (!candidate->input.Allocate(config.driver, width, height,
                                     cuda::PixelFormatGpu::rgb_u8, &error) ||
          !candidate->output.Allocate(config.driver, width, height,
                                      cuda::PixelFormatGpu::rgb_u8, &error)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      const bool final_resize = program.host_output.width != image.width ||
                                program.host_output.height != image.height;
      if ((final_resize || replace) &&
          !cuda::kernels::PrepareResizeBilinearKernel(
              *config.driver,
              reinterpret_cast<cuda::CUcontext>(
                  static_cast<std::uintptr_t>(context->retained_context_id)),
              candidate->stream,
              &candidate->resize, &error)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::capability_evidence_incomplete);
      }
      if (final_resize &&
          (!candidate->final_output.Allocate(
               config.driver, static_cast<int>(program.host_output.width),
               static_cast<int>(program.host_output.height),
               cuda::PixelFormatGpu::rgb_u8, &error) ||
           !cuda::kernels::LaunchResizeBilinearPrepared(
               candidate->resize, candidate->input, candidate->final_output,
               &error) ||
           !config.driver->StreamSynchronize(candidate->stream, nullptr))) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::capability_evidence_incomplete);
      }
      const bool blur = program.stages[0].semantic ==
                        SemanticEffectId::virtual_background_blur;
      if (blur && (!candidate->blur_tmp.Allocate(config.driver, width, height,
                                                 cuda::PixelFormatGpu::rgb_u8,
                                                 &error) ||
                   !candidate->blurred.Allocate(config.driver, width, height,
                                                cuda::PixelFormatGpu::rgb_u8,
                                                &error))) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      if (replace &&
          (!candidate->replacement_source.Allocate(
               config.driver, config.replacement_width,
               config.replacement_height, cuda::PixelFormatGpu::rgb_u8,
               &error) ||
           !candidate->replacement_frame.Allocate(config.driver, width, height,
                                                  cuda::PixelFormatGpu::rgb_u8,
                                                  &error) ||
           !candidate->replacement_source.UploadFromCpuRgb24(
               config.driver, config.replacement_rgb.data(),
               static_cast<std::size_t>(config.replacement_width) * 3u,
               candidate->stream, &error) ||
           !cuda::kernels::LaunchResizeBilinearPrepared(
               candidate->resize, candidate->replacement_source,
               candidate->replacement_frame, &error))) {
        if (!config.driver->StreamSynchronize(candidate->stream, nullptr)) {
          candidate->setup_completion_unproven = true;
          candidate->retired = true;
          *destination = std::move(candidate);
          return SetupFailure(RuntimeStatus::completion_pending,
                              ReasonId::synchronization_failed);
        }
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::capability_evidence_incomplete);
      }
      if (replace &&
          !config.driver->StreamSynchronize(candidate->stream, nullptr)) {
        candidate->setup_completion_unproven = true;
        candidate->retired = true;
        *destination = std::move(candidate);
        return SetupFailure(RuntimeStatus::completion_pending,
                            ReasonId::synchronization_failed);
      }
      if (!candidate->alpha.AllocateNchwF32(config.driver, 1, 1, model_h,
                                            model_w, &error) ||
          !candidate->alpha_resized.Allocate(config.driver, width, height,
                                             cuda::PixelFormatGpu::f32_1,
                                             &error)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      if (feather &&
          (!candidate->alpha_tmp.Allocate(config.driver, width, height,
                                          cuda::PixelFormatGpu::f32_1,
                                          &error) ||
           !candidate->alpha_feather.Allocate(config.driver, width, height,
                                              cuda::PixelFormatGpu::f32_1,
                                              &error))) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      candidate->alpha_image.ptr = candidate->alpha.ptr;
      candidate->alpha_image.pitch =
          static_cast<std::size_t>(model_w) * sizeof(float);
      candidate->alpha_image.w = model_w;
      candidate->alpha_image.h = model_h;
      candidate->alpha_image.format = cuda::PixelFormatGpu::f32_1;

      OpenCudaMattingSession::Options session_options;
      session_options.device_id = config.cuda_device_id;
      session_options.enable_tensorrt = false;
      session_options.strict_cuda_provider = true;
      candidate->session = std::make_unique<OpenCudaMattingSession>(
          config.driver, candidate->model_pack, session_options);
      ++telemetry_.allocation_requests;
      ++telemetry_.session_constructions;
      if (!candidate->session->Prepare(width, height, candidate->stream,
                                       &candidate->alpha, &error)) {
        // Session setup performs real warmup work. A failed final
        // synchronization cannot prove that the candidate's ORT/device
        // resources are idle, so retain the whole candidate instead of
        // destroying storage underneath possible work.
        if (!config.driver->StreamSynchronize(candidate->stream, nullptr)) {
          candidate->setup_completion_unproven = true;
          candidate->retired = true;
          *destination = std::move(candidate);
          return SetupFailure(RuntimeStatus::completion_pending,
                              ReasonId::synchronization_failed);
        }
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::semantic_provider_unavailable);
      }
      const auto *session_evidence = candidate->session->preparation_evidence();
      candidate->matting_evidence = candidate->session->prepared_run_evidence();
      const auto input_view = candidate->session->prepared_input_tensor_view();
      if (!session_evidence || !input_view.Valid() ||
          !candidate->matting_evidence.Valid() ||
          session_evidence->schema_version !=
              OpenCudaMattingSession::PreparationEvidence::kSchemaVersion ||
          !session_evidence->ort_user_stream_bound ||
          session_evidence->explicit_stream_synchronization ||
          !session_evidence->inference_warmup_complete ||
          session_evidence->input_tensor_active_bytes != input_view.bytes ||
          session_evidence->input_tensor_allocation_pitch_bytes !=
              input_view.allocation_pitch_bytes ||
          session_evidence->input_tensor_allocation_capacity_bytes !=
              input_view.allocation_capacity_bytes ||
          session_evidence->model_artifact != config.model_artifact ||
          candidate->matting_evidence.artifact != config.model_artifact ||
          candidate->matting_evidence.stream_identity !=
              reinterpret_cast<std::uintptr_t>(candidate->stream)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::stale_artifact,
                            ReasonId::capability_evidence_incomplete);
      }
      candidate->input_tensor.ptr = input_view.ptr;
      candidate->input_tensor.pitch = input_view.allocation_pitch_bytes;
      candidate->input_tensor.bytes = input_view.bytes;
      candidate->input_tensor.n = input_view.n;
      candidate->input_tensor.c = input_view.c;
      candidate->input_tensor.h = input_view.h;
      candidate->input_tensor.w = input_view.w;

      candidate->preprocess_spec = PreprocessSpec(candidate->model_pack);
      candidate->preprocess_binding.driver = config.driver;
      candidate->preprocess_binding.context_identity = *context;
      candidate->preprocess_binding.stream = candidate->stream;
      candidate->preprocess_binding.stream_identity = config.stream_identity;
      candidate->preprocess_binding.configuration_generation =
          config.configuration_generation;
      candidate->preprocess_binding.model_preprocess_generation =
          config.model_preprocess_generation;
      if (!candidate->preprocess.Prepare(
              candidate->preprocess_binding, candidate->preprocess_spec,
              cuda::kernels::PreparedPreprocessKernelBundle::
                  RequiredResourceBytes(),
              &error)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::capability_evidence_incomplete);
      }

      candidate->postprocess_binding.driver = config.driver;
      candidate->postprocess_binding.context_identity = *context;
      candidate->postprocess_binding.stream = candidate->stream;
      candidate->postprocess_binding.stream_identity = config.stream_identity;
      candidate->postprocess_binding.configuration_generation =
          config.configuration_generation;
      if (!candidate->postprocess.Prepare(
              candidate->postprocess_binding,
              cuda::kernels::PreparedOpenCudaVbKernelBundle::
                  RequiredResourceBytes(),
              &error)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::capability_evidence_incomplete);
      }
      if (program.stages[program.stage_count - 1].semantic ==
              SemanticEffectId::vignette &&
          !cuda::kernels::PrepareVignetteKernel(
              *config.driver,
              reinterpret_cast<cuda::CUcontext>(static_cast<std::uintptr_t>(
                  context->retained_context_id)), candidate->stream,
              &candidate->vignette, &error)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::not_prepared,
                            ReasonId::capability_evidence_incomplete);
      }

      cuda::CudaPreparedTransferCompletion::PrepareOptions transfer_options;
      transfer_options.binding.context = *context;
      transfer_options.binding.stream = candidate->stream;
      transfer_options.binding.stream_identity = config.stream_identity;
      transfer_options.binding.owner_identity = config.adapter_instance_id;
      transfer_options.binding.configuration_generation =
          config.configuration_generation;
      transfer_options.upload_row_bytes = static_cast<std::size_t>(
          program.host_input.planes[0].row_stride_bytes);
      transfer_options.upload_rows = program.host_input.height;
      transfer_options.readback_row_bytes = static_cast<std::size_t>(
          program.host_output.planes[0].row_stride_bytes);
      transfer_options.readback_rows = program.host_output.height;
      transfer_options.timed_stage_count =
          static_cast<std::uint32_t>(program.stage_count);
      std::uint64_t pinned_capacity = 0;
      if (!CheckedAdd(program.host_input.planes[0].size_bytes,
                      program.host_output.planes[0].size_bytes,
                      &pinned_capacity) ||
          pinned_capacity > std::numeric_limits<std::size_t>::max()) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      transfer_options.budget.max_pinned_host_bytes =
          static_cast<std::size_t>(pinned_capacity);
      transfer_options.budget.max_events =
          static_cast<std::uint32_t>(program.stage_count * 2u + 1u);
      const auto transfer_prepare =
          candidate->transfer.Prepare(config.driver, transfer_options);
      if (!transfer_prepare.ok()) {
        CleanupCandidate(candidate.get());
        return SetupFailure(TransferStatus(transfer_prepare.status),
                            HalReason(TransferStatus(transfer_prepare.status)));
      }
      candidate->transfer_binding = transfer_options.binding;

      PreparedOpenCudaVbPlanInput exact_plan_input = preflight;
      exact_plan_input.input_pitch_bytes = candidate->input.pitch;
      exact_plan_input.output_pitch_bytes = candidate->output.pitch;
      exact_plan_input.final_output_pitch_bytes =
          candidate->final_output.Valid() ? candidate->final_output.pitch
                                          : candidate->output.pitch;
      if (!CheckedMul(candidate->alpha.pitch,
                      static_cast<std::uint64_t>(model_h),
                      &exact_plan_input.model_alpha_capacity_bytes)) {
        CleanupCandidate(candidate.get());
        return SetupFailure(RuntimeStatus::resource_exhausted,
                            ReasonId::resource_exhausted);
      }
      exact_plan_input.input_tensor_capacity_bytes =
          input_view.allocation_capacity_bytes;
      exact_plan_input.resized_alpha_pitch_bytes =
          candidate->alpha_resized.pitch;
      exact_plan_input.alpha_scratch_pitch_bytes = candidate->alpha_tmp.pitch;
      exact_plan_input.alpha_feather_pitch_bytes =
          candidate->alpha_feather.pitch;
      exact_plan_input.blur_scratch_pitch_bytes = candidate->blur_tmp.pitch;
      exact_plan_input.blurred_pitch_bytes = candidate->blurred.pitch;
      exact_plan_input.replacement_source_pitch_bytes =
          candidate->replacement_source.pitch;
      exact_plan_input.replacement_source_height =
          static_cast<std::uint64_t>(candidate->replacement_source.h);
      exact_plan_input.replacement_frame_pitch_bytes =
          candidate->replacement_frame.pitch;
      for (std::size_t index = 0; index < program.stage_count; ++index) {
        if (program.stages[index].semantic == SemanticEffectId::vignette)
          continue;
        exact_plan_input.semantic = program.stages[index].semantic;
        candidate->postprocess_plans[index] =
            BuildPreparedOpenCudaVirtualBackgroundPlan(exact_plan_input);
        if (!candidate->postprocess_plans[index].ready()) {
          const auto failed = candidate->postprocess_plans[index];
          CleanupCandidate(candidate.get());
          return SetupFailure(failed.status, failed.reason);
        }
        if (!PreparePostprocessLaunches(candidate.get(), index)) {
          CleanupCandidate(candidate.get());
          return SetupFailure(RuntimeStatus::not_prepared,
                              ReasonId::capability_evidence_incomplete);
        }
      }
      const hal::ResourceBudget required =
          candidate->postprocess_plans[0].required_budget;
      candidate->reserved_budget = required;
      candidate->handle.handle_id = next_handle_id_++;
      if (candidate->handle.handle_id == 0)
        candidate->handle.handle_id = next_handle_id_++;
      candidate->handle.program_id = program.program_id;
      candidate->handle.invalidation_key = program.invalidation_key;
      candidate->handle.execution_domain = program.execution_domain;
      candidate->handle.opaque[0] =
          reinterpret_cast<std::uintptr_t>(candidate->stream);
      candidate->handle.opaque[1] =
          candidate->matting_evidence.preparation_generation;

      result.status = RuntimeStatus::ready;
      result.reason = ReasonId::none;
      result.handle = candidate->handle;
      result.evidence.api_version = kPreparedOpenCudaRuntimeApiVersion;
      result.evidence.status = RuntimeStatus::ready;
      result.evidence.semantic = program.stages[0].semantic;
      result.evidence.provider =
          program.stages[program.stage_count - 1].semantic ==
                  SemanticEffectId::vignette
              ? SemanticEffectProviderId::invalid
              : kOrtProvider;
      result.evidence.execution_domain = program.execution_domain;
      result.evidence.model_artifact = config.model_artifact;
      result.evidence.reserved_budget = required;
      result.evidence.geometry_plan = candidate->postprocess_plans[0];
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
      *destination = std::move(candidate);
      return result;
    } catch (...) {
      return SetupFailure(RuntimeStatus::resource_exhausted,
                          ReasonId::resource_exhausted);
    }
  }

  PreparedOpenCudaRuntimeFrameResult
  ExecuteFrame(const PreparedOpenCudaRuntimeHandle &handle,
               const PreparedOpenCudaRuntimeFrame &frame) noexcept override {
    PreparedOpenCudaRuntimeFrameResult result;
    result.reason = ReasonId::none;
    Slot *slot = Find(handle);
    if (!slot) {
      result.status = RuntimeStatus::foreign_handle;
      result.reason = ReasonId::stale_program;
      return result;
    }
    const std::uint64_t frame_started_ns =
        MonotonicNowNanoseconds(slot->config);
    const auto finish = [&](PreparedOpenCudaRuntimeFrameResult value) noexcept {
      FinishFrameTiming(slot->config, frame_started_ns, &value);
      return value;
    };
    if (slot->quarantined) {
      result.status = RuntimeStatus::quarantined;
      result.reason = ReasonId::device_lost;
      return finish(result);
    }
    if (frame.api_version != kPreparedOpenCudaRuntimeApiVersion ||
        !frame.input || !frame.output || frame.capture_sequence == 0 ||
        frame.input_row_stride_bytes !=
            slot->program.host_input.planes[0].row_stride_bytes ||
        frame.output_row_stride_bytes !=
            slot->program.host_output.planes[0].row_stride_bytes ||
        frame.input_span_bytes <
            slot->program.host_input.planes[0].size_bytes ||
        frame.output_span_bytes <
            slot->program.host_output.planes[0].size_bytes) {
      result.status = RuntimeStatus::host_binding_invalid;
      result.reason = ReasonId::host_frame_binding_invalid;
      return finish(result);
    }
    if (slot->transfer.frame_reuse_blocked()) {
      result.status = RuntimeStatus::completion_pending;
      result.reason = ReasonId::resource_exhausted;
      return finish(result);
    }
    if (slot->config.wait_policy.stop_requested &&
        slot->config.wait_policy.stop_requested(
            slot->config.wait_policy.callback_context)) {
      result.status = RuntimeStatus::stop_requested;
      result.reason = ReasonId::shutdown_requested;
      return finish(result);
    }
    if (slot->config.wait_policy.cancel_requested &&
        slot->config.wait_policy.cancel_requested(
            slot->config.wait_policy.callback_context)) {
      result.status = RuntimeStatus::cancelled;
      result.reason = ReasonId::shutdown_requested;
      return finish(result);
    }
    if (slot->config.driver->BindPreparedContext(slot->context) !=
        cuda::CUDA_SUCCESS) {
      result.status = RuntimeStatus::stale_generation;
      result.reason = ReasonId::wrong_generation;
      return finish(result);
    }

    const auto begin = slot->transfer.BeginFrame(slot->transfer_binding,
                                                 frame.capture_sequence);
    if (!begin.ok())
      return finish(TransferFailure(slot, begin.status, {},
                                    PreparedOpenCudaOperationId::upload));
    std::memcpy(slot->transfer.upload_data(), frame.input,
                slot->transfer.upload_size_bytes());
    const auto upload = slot->transfer.EnqueueUpload(
        slot->transfer_binding, slot->input.ptr, slot->input.pitch);
    if (!upload.ok())
      return finish(TransferFailure(slot, upload.status, {},
                                    PreparedOpenCudaOperationId::upload));
    result.transfers.uploads = 1;

    RuntimeStatus analysis_status = RuntimeStatus::executed;
    bool analysis_ran = false;
    bool inference_only = false;
    const std::uint32_t disabled_mask =
        frame.disabled_stage_mask | (frame.stage_disabled ? 1u : 0u);
    bool model_stage_enabled = false;
    for (std::size_t index = 0; index < slot->program.stage_count; ++index)
      model_stage_enabled |= UsesMatting(slot->program.stages[index].semantic) &&
          (disabled_mask & (std::uint32_t{1} << index)) == 0;
    if (model_stage_enabled) {
      analysis_ran = true;
      cuda::kernels::PreparedPreprocessKernelStatus preprocess_status{};
      if (!slot->preprocess.Execute(slot->preprocess_binding,
                                    slot->preprocess_spec, slot->input,
                                    slot->input_tensor, &preprocess_status)) {
        analysis_status =
            preprocess_status ==
                    cuda::kernels::PreparedPreprocessKernelStatus::device_lost
                ? RuntimeStatus::device_lost
                : RuntimeStatus::preprocess_failed;
      } else {
        inference_only = true;
        const bool cancelled = slot->config.wait_policy.cancel_requested &&
                               slot->config.wait_policy.cancel_requested(
                                   slot->config.wait_policy.callback_context);
        const auto inference = slot->session->RunPreparedInferenceTyped(
            slot->stream, &slot->alpha, slot->matting_evidence, cancelled);
        if (!inference.ok()) {
          if (inference.status ==
              MattingPreparedRunStatus::cancelled_or_shutdown) {
            analysis_status = cancelled ? RuntimeStatus::cancelled
                                        : RuntimeStatus::stop_requested;
          } else if (inference.quarantine !=
                     MattingPreparedRunQuarantine::none) {
            analysis_status = RuntimeStatus::device_lost;
          } else {
            analysis_status = RuntimeStatus::inference_failed;
          }
        }
      }
    }
    if (analysis_status == RuntimeStatus::device_lost) {
      slot->quarantined = true;
      result.status = RuntimeStatus::device_lost;
      result.reason = ReasonId::device_lost;
      result.stage_failure_status = analysis_status;
      result.stage_failure_reason = ReasonId::device_lost;
      result.failed_operation = PreparedOpenCudaOperationId::inference;
      result.analysis_ran = analysis_ran;
      result.inference_only_path = inference_only;
      return finish(result);
    }
    if (analysis_status == RuntimeStatus::cancelled ||
        analysis_status == RuntimeStatus::stop_requested) {
      result.status = analysis_status;
      result.reason = HalReason(analysis_status);
      result.stage_failure_status = analysis_status;
      result.stage_failure_reason = result.reason;
      result.failed_operation = PreparedOpenCudaOperationId::inference;
      result.analysis_ran = analysis_ran;
      result.inference_only_path = inference_only;
      return finish(result);
    }
    bool current_is_output = false;
    bool required_failed = false;
    bool optional_failed = false;
    for (std::size_t index = 0; index < slot->program.stage_count; ++index) {
      auto &stage_result = result.stages[index];
      const auto &stage = slot->program.stages[index];
      stage_result.stage_id = stage.stage_id;
      stage_result.semantic = stage.semantic;
      const bool disabled = (disabled_mask & (std::uint32_t{1} << index)) != 0;
      const auto stage_index = static_cast<std::uint32_t>(index);
      const auto stage_begin =
          slot->transfer.RecordStageBegin(slot->transfer_binding, stage_index);
      if (!stage_begin.ok())
        return finish(
            TransferFailure(slot, stage_begin.status, result.transfers,
                            PreparedOpenCudaOperationId::completion_wait));
      RuntimeStatus stage_status = analysis_status;
      PreparedOpenCudaOperationId failed_operation =
          analysis_status == RuntimeStatus::preprocess_failed
              ? PreparedOpenCudaOperationId::preprocess
              : PreparedOpenCudaOperationId::inference;
      if (disabled) {
        stage_result.status = StageExecutionStatus::disabled_preserved;
        stage_result.reason = ReasonId::circuit_breaker_open;
        stage_result.valid_output_preserved = true;
      } else if (stage.semantic == SemanticEffectId::vignette) {
        // Terminal fixed-center vignette is launched after any final resize.
        stage_result.status = StageExecutionStatus::succeeded;
      } else if (analysis_status == RuntimeStatus::executed) {
        const auto &current = current_is_output ? slot->output : slot->input;
        const auto &next = current_is_output ? slot->input : slot->output;
        slot->input_ptr_arg = current.ptr;
        slot->input_pitch_arg = static_cast<std::uint32_t>(current.pitch);
        slot->output_ptr_arg = next.ptr;
        slot->output_pitch_arg = static_cast<std::uint32_t>(next.pitch);
        stage_status = LaunchPostprocess(*slot, index, &failed_operation);
        if (stage_status == RuntimeStatus::device_lost) {
          slot->quarantined = true;
          result.status = RuntimeStatus::device_lost;
          result.reason = ReasonId::device_lost;
          result.stage_failure_status = stage_status;
          result.stage_failure_reason = ReasonId::device_lost;
          result.failed_operation = failed_operation;
          result.analysis_ran = analysis_ran;
          result.inference_only_path = inference_only;
          return finish(result);
        }
        if (stage_status == RuntimeStatus::executed) {
          current_is_output = !current_is_output;
          stage_result.status = StageExecutionStatus::succeeded;
        }
      }
      if (!disabled && stage_status != RuntimeStatus::executed) {
        stage_result.reason = HalReason(stage_status);
        stage_result.valid_output_preserved = !stage.required;
        if (stage.required) {
          stage_result.status = StageExecutionStatus::required_failed;
          required_failed = true;
        } else {
          stage_result.status = StageExecutionStatus::optional_failed_open;
          optional_failed = true;
        }
        result.stage_failure_status = stage_status;
        result.stage_failure_reason = stage_result.reason;
        result.failed_operation = failed_operation;
      }
      const auto stage_end =
          slot->transfer.RecordStageEnd(slot->transfer_binding, stage_index);
      if (!stage_end.ok())
        return finish(
            TransferFailure(slot, stage_end.status, result.transfers,
                            PreparedOpenCudaOperationId::completion_wait));
    }
    result.stage_count = slot->program.stage_count;
    const cuda::CudaImage *readback_source =
        current_is_output ? &slot->output : &slot->input;
    if (slot->final_output.Valid()) {
      std::string resize_error;
      if (!cuda::kernels::LaunchResizeBilinearPrepared(
              slot->resize, *readback_source, slot->final_output,
              &resize_error)) {
        result.status = RuntimeStatus::postprocess_failed;
        result.reason = ReasonId::optional_stage_failed_open;
        result.failed_operation = PreparedOpenCudaOperationId::final_resize;
        return finish(result);
      }
      readback_source = &slot->final_output;
    }
    const std::size_t vignette_index = slot->program.stage_count - 1;
    if (slot->program.stages[vignette_index].semantic == SemanticEffectId::vignette &&
        (disabled_mask & (std::uint32_t{1} << vignette_index)) == 0 &&
        !required_failed) {
      std::string vignette_error;
      const float intensity = std::clamp(
          static_cast<float>(slot->config.vignette_intensity) / 100.0f, 0.0f, 1.0f);
      if (!cuda::kernels::LaunchVignettePrepared(
              slot->vignette, *readback_source, *readback_source, intensity,
              static_cast<float>(readback_source->w) * 0.5f,
              static_cast<float>(readback_source->h) * 0.5f,
              &vignette_error)) {
        auto &stage_result = result.stages[vignette_index];
        stage_result.reason = ReasonId::optional_stage_failed_open;
        result.stage_failure_status = RuntimeStatus::postprocess_failed;
        result.stage_failure_reason = stage_result.reason;
        result.failed_operation = PreparedOpenCudaOperationId::vignette;
        if (slot->program.stages[vignette_index].required) {
          stage_result.status = StageExecutionStatus::required_failed;
          stage_result.valid_output_preserved = false;
          required_failed = true;
        } else {
          stage_result.status = StageExecutionStatus::optional_failed_open;
          stage_result.valid_output_preserved = true;
          optional_failed = true;
        }
      }
    }
    const auto readback = slot->transfer.EnqueueReadback(
        slot->transfer_binding, readback_source->ptr, readback_source->pitch);
    if (!readback.ok())
      return finish(
          TransferFailure(slot, readback.status, result.transfers,
                          PreparedOpenCudaOperationId::final_readback));
    result.transfers.final_readbacks = 1;
    const auto completion =
        slot->transfer.RecordCompletion(slot->transfer_binding);
    if (!completion.ok())
      return finish(
          TransferFailure(slot, completion.status, result.transfers,
                          PreparedOpenCudaOperationId::completion_wait));
    const auto wait = slot->transfer.WaitForCompletion(
        slot->transfer_binding, slot->config.wait_policy);
    result.transfers.synchronizations = 1;
    if (!wait.ok()) {
      auto failed =
          TransferFailure(slot, wait.status, result.transfers,
                          PreparedOpenCudaOperationId::completion_wait);
      failed.analysis_ran = analysis_ran && frame.stage_required;
      failed.inference_only_path = inference_only;
      failed.stage_failure_status = TransferStatus(wait.status);
      failed.stage_failure_reason = HalReason(failed.stage_failure_status);
      failed.failed_operation = PreparedOpenCudaOperationId::completion_wait;
      return finish(failed);
    }

    for (std::size_t index = 0; index < result.stage_count; ++index) {
      result.stages[index].elapsed_ns = slot->transfer.stage_elapsed_nanoseconds(
          static_cast<std::uint32_t>(index));
    }

    result.stage_elapsed_ns = slot->transfer.stage_elapsed_nanoseconds(0);
    const bool analysis_completed =
        analysis_ran && analysis_status != RuntimeStatus::cancelled &&
        analysis_status != RuntimeStatus::stop_requested;
    result.analysis_ran = analysis_completed;
    result.inference_only_path = inference_only;
    if (!required_failed) {
      std::memcpy(frame.output, slot->transfer.readback_data(),
                  slot->transfer.readback_size_bytes());
      result.transfers.cpu_tail_boundaries = 1;
      result.output_valid = true;
      result.valid_input_preserved = optional_failed || !current_is_output;
      result.status = optional_failed ? RuntimeStatus::optional_failed_open
                                      : RuntimeStatus::executed;
      result.reason = optional_failed ? ReasonId::optional_stage_failed_open
                                      : ReasonId::none;
      return finish(result);
    }
    if (analysis_status == RuntimeStatus::cancelled ||
        analysis_status == RuntimeStatus::stop_requested) {
      result.status = analysis_status;
      result.reason = HalReason(analysis_status);
      return finish(result);
    }
    result.status = RuntimeStatus::required_failed;
    result.reason = ReasonId::required_stage_failed;
    return finish(result);
  }

  PreparedOpenCudaRuntimeStatus
  ResetTemporalState(const PreparedOpenCudaRuntimeHandle &handle,
                     hal::ReasonId) noexcept override {
    Slot *slot = Find(handle);
    if (!slot)
      return RuntimeStatus::foreign_handle;
    return slot->quarantined ? RuntimeStatus::quarantined
                             : RuntimeStatus::ready;
  }

  PreparedOpenCudaRuntimeStatus
  DestroyPrepared(PreparedOpenCudaRuntimeHandle *handle) noexcept override {
    if (!handle)
      return RuntimeStatus::foreign_handle;
    for (auto &slot : slots_) {
      if (!slot || !SameRuntimeHandle(slot->handle, *handle))
        continue;
      const auto status = CleanupSlot(slot.get());
      *handle = {};
      if (status == RuntimeStatus::completion_pending) {
        slot->retired = true;
        return status;
      }
      slot.reset();
      return status;
    }
    *handle = {};
    return RuntimeStatus::foreign_handle;
  }

  PreparedOpenCudaShutdownSnapshot RequestShutdown() noexcept override {
    shutdown_ = true;
    ++shutdown_snapshot_.shutdown_attempts;
    shutdown_snapshot_.shutdown_requested = true;
    for (auto &slot : slots_) {
      if (!slot)
        continue;
      const auto status = CleanupSlot(slot.get());
      if (status != RuntimeStatus::completion_pending)
        slot.reset();
      else
        slot->retired = true;
    }
    const auto pending_count = static_cast<std::uint32_t>(
        std::count_if(slots_.begin(), slots_.end(),
                      [](const auto &slot) { return slot != nullptr; }));
    const bool pending = pending_count != 0;
    if (!pending && owns_stream_ && owned_stream_ && owned_driver_) {
      if (owned_driver_->DestroyStream(owned_stream_, nullptr)) {
        owned_stream_ = nullptr;
        owns_stream_ = false;
        shutdown_snapshot_.stream_destroyed = true;
        ++shutdown_snapshot_.stream_destructions;
      }
    }
    shutdown_snapshot_.pending_prepared_slots = pending_count;
    shutdown_snapshot_.stream_owned = stream_was_owned_;
    shutdown_snapshot_.quiescent =
        !pending && (!stream_was_owned_ || shutdown_snapshot_.stream_destroyed);
    shutdown_snapshot_.driver_context_release_permitted =
        shutdown_snapshot_.quiescent;
    shutdown_snapshot_.status = shutdown_snapshot_.quiescent
                                    ? RuntimeStatus::ready
                                    : RuntimeStatus::completion_pending;
    return shutdown_snapshot_;
  }

  PreparedOpenCudaShutdownSnapshot ShutdownSnapshot() const noexcept override {
    return shutdown_snapshot_;
  }

  PreparedOpenCudaRuntimeTelemetry Telemetry() const noexcept override {
    return telemetry_;
  }

private:
  struct Slot {
    struct PreparedLaunch {
      PreparedOpenCudaOperationId operation =
          PreparedOpenCudaOperationId::invalid;
      cuda::kernels::PreparedOpenCudaVbLaunch request{};
      std::array<void *, 13> parameters{};
    };

    hal::PreparedProgram program{};
    PreparedOpenCudaProductionConfig config{};
    open_video::ModelPack model_pack{};
    PreparedOpenCudaRuntimeHandle handle{};
    hal::ResourceBudget reserved_budget{};
    cuda::CudaContextIdentity context{};
    cuda::CUstream stream = nullptr;
    cuda::CudaImage input{};
    cuda::CudaImage output{};
    cuda::CudaImage final_output{};
    cuda::kernels::PreparedResizeBilinearKernel resize{};
    cuda::kernels::PreparedVignetteKernel vignette{};
    cuda::CudaImage blur_tmp{};
    cuda::CudaImage blurred{};
    cuda::CudaImage replacement_source{};
    cuda::CudaImage replacement_frame{};
    cuda::CudaTensor alpha{};
    cuda::CudaImage alpha_image{};
    cuda::CudaImage alpha_resized{};
    cuda::CudaImage alpha_tmp{};
    cuda::CudaImage alpha_feather{};
    cuda::CudaTensor input_tensor{};
    std::unique_ptr<OpenCudaMattingSession> session{};
    MattingPreparedRunEvidence matting_evidence{};
    cuda::kernels::ModelPreprocessSpec preprocess_spec{};
    cuda::kernels::PreparedPreprocessBinding preprocess_binding{};
    cuda::kernels::PreparedPreprocessKernelBundle preprocess{};
    cuda::kernels::PreparedOpenCudaVbBinding postprocess_binding{};
    cuda::kernels::PreparedOpenCudaVbKernelBundle postprocess{};
    std::array<PreparedOpenCudaVbPlan, hal::kMaxPreparedStages> postprocess_plans{};
    std::array<std::array<PreparedLaunch, PreparedOpenCudaVbPlan::kMaxLaunches>,
               hal::kMaxPreparedStages>
        postprocess_launches{};
    unsigned long long input_ptr_arg = 0;
    unsigned long long output_ptr_arg = 0;
    unsigned long long model_alpha_ptr_arg = 0;
    unsigned long long resized_alpha_ptr_arg = 0;
    unsigned long long alpha_tmp_ptr_arg = 0;
    unsigned long long alpha_feather_ptr_arg = 0;
    unsigned long long blur_tmp_ptr_arg = 0;
    unsigned long long blurred_ptr_arg = 0;
    unsigned long long replacement_frame_ptr_arg = 0;
    std::uint32_t input_pitch_arg = 0;
    std::uint32_t output_pitch_arg = 0;
    std::uint32_t model_alpha_pitch_arg = 0;
    std::uint32_t resized_alpha_pitch_arg = 0;
    std::uint32_t alpha_tmp_pitch_arg = 0;
    std::uint32_t alpha_feather_pitch_arg = 0;
    std::uint32_t blur_tmp_pitch_arg = 0;
    std::uint32_t blurred_pitch_arg = 0;
    std::uint32_t replacement_frame_pitch_arg = 0;
    std::uint32_t frame_width_arg = 0;
    std::uint32_t frame_height_arg = 0;
    std::uint32_t model_width_arg = 0;
    std::uint32_t model_height_arg = 0;
    int strength_arg = 0;
    int alpha_feather_radius_arg = 0;
    std::array<std::uint8_t, 3> remove_background_rgb_arg{};
    float key_light_r_arg = 255.0f;
    float key_light_g_arg = 255.0f;
    float key_light_b_arg = 255.0f;
    float key_light_intensity_arg = 0.5f;
    float key_light_direction_arg = 0.0f;
    cuda::CudaPreparedTransferCompletion::Binding transfer_binding{};
    cuda::CudaPreparedTransferCompletion transfer{};
    bool quarantined = false;
    bool retired = false;
    bool setup_completion_unproven = false;
  };

  static PreparedOpenCudaRuntimePrepareResult
  SetupFailure(RuntimeStatus status, ReasonId reason) noexcept {
    PreparedOpenCudaRuntimePrepareResult result;
    result.status = status;
    result.reason = reason;
    return result;
  }

  Slot *Find(const PreparedOpenCudaRuntimeHandle &handle) noexcept {
    for (auto &slot : slots_) {
      if (slot && !slot->retired && SameRuntimeHandle(slot->handle, handle)) {
        return slot.get();
      }
    }
    return nullptr;
  }

  static void FreeImage(cuda::CudaDriverApi *driver,
                        cuda::CudaImage *image) noexcept {
    if (image && image->ptr != 0 && image->owns_memory) {
      (void)image->Free(driver, nullptr);
    }
  }

  static void FreeTensor(cuda::CudaDriverApi *driver,
                         cuda::CudaTensor *tensor) noexcept {
    if (tensor && tensor->ptr != 0 && tensor->owns_memory) {
      (void)tensor->Free(driver, nullptr);
    }
  }

  static void CleanupCandidate(Slot *slot) noexcept {
    if (!slot)
      return;
    slot->session.reset();
    slot->preprocess.Shutdown();
    slot->postprocess.Shutdown();
    (void)slot->transfer.Shutdown();
    std::string resize_error;
    (void)slot->config.driver->BindPreparedContext(slot->context);
    (void)cuda::kernels::DestroyVignetteKernelPrepared(&slot->vignette,
                                                        &resize_error);
    (void)cuda::kernels::DestroyResizeBilinearKernelPrepared(&slot->resize,
                                                              &resize_error);
    FreeTensor(slot->config.driver, &slot->alpha);
    FreeImage(slot->config.driver, &slot->alpha_feather);
    FreeImage(slot->config.driver, &slot->alpha_tmp);
    FreeImage(slot->config.driver, &slot->alpha_resized);
    FreeImage(slot->config.driver, &slot->blurred);
    FreeImage(slot->config.driver, &slot->blur_tmp);
    FreeImage(slot->config.driver, &slot->replacement_frame);
    FreeImage(slot->config.driver, &slot->replacement_source);
    FreeImage(slot->config.driver, &slot->output);
    FreeImage(slot->config.driver, &slot->final_output);
    FreeImage(slot->config.driver, &slot->input);
    slot->stream = nullptr;
  }

  static RuntimeStatus CleanupSlot(Slot *slot) noexcept {
    if (!slot)
      return RuntimeStatus::ready;
    if (slot->setup_completion_unproven)
      return RuntimeStatus::completion_pending;
    if (slot->session)
      slot->session->RequestShutdown();
    const auto transfer = slot->transfer.Shutdown();
    if (transfer.status ==
        cuda::CudaPreparedTransferCompletion::Status::resource_exhausted) {
      return RuntimeStatus::completion_pending;
    }
    slot->session.reset();
    slot->preprocess.Shutdown();
    slot->postprocess.Shutdown();
    std::string resize_error;
    if (slot->config.driver->BindPreparedContext(slot->context) !=
        cuda::CUDA_SUCCESS) {
      return RuntimeStatus::completion_pending;
    }
    if (!cuda::kernels::DestroyVignetteKernelPrepared(&slot->vignette,
                                                       &resize_error)) {
      return RuntimeStatus::completion_pending;
    }
    if (!cuda::kernels::DestroyResizeBilinearKernelPrepared(&slot->resize,
                                                             &resize_error)) {
      return RuntimeStatus::completion_pending;
    }
    FreeTensor(slot->config.driver, &slot->alpha);
    FreeImage(slot->config.driver, &slot->alpha_feather);
    FreeImage(slot->config.driver, &slot->alpha_tmp);
    FreeImage(slot->config.driver, &slot->alpha_resized);
    FreeImage(slot->config.driver, &slot->blurred);
    FreeImage(slot->config.driver, &slot->blur_tmp);
    FreeImage(slot->config.driver, &slot->replacement_frame);
    FreeImage(slot->config.driver, &slot->replacement_source);
    FreeImage(slot->config.driver, &slot->output);
    FreeImage(slot->config.driver, &slot->final_output);
    FreeImage(slot->config.driver, &slot->input);
    slot->stream = nullptr;
    return transfer.ok() ? RuntimeStatus::ready
                         : RuntimeStatus::transfer_failed;
  }

  static PreparedOpenCudaRuntimeFrameResult
  TransferFailure(Slot *slot,
                  cuda::CudaPreparedTransferCompletion::Status status,
                  hal::TransferCounters counters = {},
                  PreparedOpenCudaOperationId failed_operation =
                      PreparedOpenCudaOperationId::invalid) noexcept {
    PreparedOpenCudaRuntimeFrameResult result;
    result.status = TransferStatus(status);
    result.reason = HalReason(result.status);
    result.stage_failure_status = result.status;
    result.stage_failure_reason = result.reason;
    result.failed_operation = failed_operation;
    result.transfers = counters;
    if (result.status == RuntimeStatus::device_lost && slot)
      slot->quarantined = true;
    return result;
  }

  static bool PreparePostprocessLaunches(Slot *slot,
                                         std::size_t stage_index) noexcept {
    using cuda::kernels::OpenCudaVbKernelId;
    if (!slot || stage_index >= slot->program.stage_count ||
        !ExactPreparedOpenCudaVirtualBackgroundPlan(
            slot->postprocess_plans[stage_index]) ||
        slot->postprocess_plans[stage_index].launch_count >
            slot->postprocess_launches[stage_index].size()) {
      return false;
    }
    const auto &plan = slot->postprocess_plans[stage_index];
    const auto fits_pitch = [](std::size_t pitch) noexcept {
      return pitch <= std::numeric_limits<std::uint32_t>::max();
    };
    const bool feather = plan.alpha_feather_radius != 0;
    const bool replace = slot->program.stages[stage_index].semantic ==
                         SemanticEffectId::virtual_background_replace;
    if (!slot->alpha_resized.Valid() ||
        (feather &&
         (!slot->alpha_tmp.Valid() || !slot->alpha_feather.Valid())) ||
        (!feather &&
         (slot->alpha_tmp.ptr != 0 || slot->alpha_tmp.pitch != 0 ||
          slot->alpha_feather.ptr != 0 || slot->alpha_feather.pitch != 0))) {
      return false;
    }
    if (!fits_pitch(slot->input.pitch) || !fits_pitch(slot->output.pitch) ||
        !fits_pitch(slot->alpha_image.pitch) ||
        !fits_pitch(slot->alpha_resized.pitch) ||
        (feather && (!fits_pitch(slot->alpha_tmp.pitch) ||
                     !fits_pitch(slot->alpha_feather.pitch))) ||
        (slot->program.stages[stage_index].semantic ==
             SemanticEffectId::virtual_background_blur &&
         (!fits_pitch(slot->blur_tmp.pitch) ||
          !fits_pitch(slot->blurred.pitch))) ||
        (replace && (!slot->replacement_source.Valid() ||
                     !slot->replacement_frame.Valid() ||
                     !fits_pitch(slot->replacement_frame.pitch)))) {
      return false;
    }

    slot->input_ptr_arg = slot->input.ptr;
    slot->output_ptr_arg = slot->output.ptr;
    slot->model_alpha_ptr_arg = slot->alpha_image.ptr;
    slot->resized_alpha_ptr_arg = slot->alpha_resized.ptr;
    slot->alpha_tmp_ptr_arg = slot->alpha_tmp.ptr;
    slot->alpha_feather_ptr_arg = slot->alpha_feather.ptr;
    slot->blur_tmp_ptr_arg = slot->blur_tmp.ptr;
    slot->blurred_ptr_arg = slot->blurred.ptr;
    slot->replacement_frame_ptr_arg = slot->replacement_frame.ptr;
    slot->input_pitch_arg = static_cast<std::uint32_t>(slot->input.pitch);
    slot->output_pitch_arg = static_cast<std::uint32_t>(slot->output.pitch);
    slot->model_alpha_pitch_arg =
        static_cast<std::uint32_t>(slot->alpha_image.pitch);
    slot->resized_alpha_pitch_arg =
        static_cast<std::uint32_t>(slot->alpha_resized.pitch);
    slot->alpha_tmp_pitch_arg =
        static_cast<std::uint32_t>(slot->alpha_tmp.pitch);
    slot->alpha_feather_pitch_arg =
        static_cast<std::uint32_t>(slot->alpha_feather.pitch);
    slot->blur_tmp_pitch_arg = static_cast<std::uint32_t>(slot->blur_tmp.pitch);
    slot->blurred_pitch_arg = static_cast<std::uint32_t>(slot->blurred.pitch);
    slot->replacement_frame_pitch_arg =
        static_cast<std::uint32_t>(slot->replacement_frame.pitch);
    slot->frame_width_arg = plan.frame_width;
    slot->frame_height_arg = plan.frame_height;
    slot->model_width_arg = plan.model_width;
    slot->model_height_arg = plan.model_height;
    slot->strength_arg = static_cast<int>(plan.strength);
    slot->alpha_feather_radius_arg =
        static_cast<int>(plan.alpha_feather_radius);
    slot->remove_background_rgb_arg = slot->config.remove_background_rgb;
    const int preset =
        std::clamp(slot->config.key_light_temperature_preset, 0, 2);
    slot->key_light_r_arg = preset == 2 ? 228.0f : 255.0f;
    slot->key_light_g_arg = preset == 0 ? 255.0f : 242.0f;
    slot->key_light_b_arg = preset == 1 ? 228.0f : 255.0f;
    slot->key_light_intensity_arg = std::clamp(
        static_cast<float>(slot->config.key_light_intensity) / 100.0f, 0.0f,
        1.0f);
    slot->key_light_direction_arg = std::clamp(
        static_cast<float>(slot->config.key_light_direction_pan_degrees) /
            90.0f,
        -1.0f, 1.0f);

    for (std::uint32_t index = 0; index < plan.launch_count; ++index) {
      const auto &geometry = plan.launches[index];
      auto &prepared = slot->postprocess_launches[stage_index][index];
      prepared = {};
      prepared.operation = geometry.operation;
      prepared.request.grid_x = geometry.grid_x;
      prepared.request.grid_y = geometry.grid_y;
      prepared.request.block_x = geometry.block_x;
      prepared.request.block_y = geometry.block_y;
      auto &parameters = prepared.parameters;
      switch (geometry.operation) {
      case PreparedOpenCudaOperationId::alpha_resize:
        prepared.request.kernel = OpenCudaVbKernelId::resize_f32;
        parameters[0] = &slot->model_alpha_ptr_arg;
        parameters[1] = &slot->model_alpha_pitch_arg;
        parameters[2] = &slot->model_width_arg;
        parameters[3] = &slot->model_height_arg;
        parameters[4] = &slot->resized_alpha_ptr_arg;
        parameters[5] = &slot->resized_alpha_pitch_arg;
        parameters[6] = &slot->frame_width_arg;
        parameters[7] = &slot->frame_height_arg;
        break;
      case PreparedOpenCudaOperationId::alpha_feather_horizontal:
        prepared.request.kernel = OpenCudaVbKernelId::blur_h_f32;
        parameters[0] = &slot->resized_alpha_ptr_arg;
        parameters[1] = &slot->resized_alpha_pitch_arg;
        parameters[2] = &slot->frame_width_arg;
        parameters[3] = &slot->frame_height_arg;
        parameters[4] = &slot->alpha_tmp_ptr_arg;
        parameters[5] = &slot->alpha_tmp_pitch_arg;
        parameters[6] = &slot->alpha_feather_radius_arg;
        break;
      case PreparedOpenCudaOperationId::alpha_feather_vertical:
        prepared.request.kernel = OpenCudaVbKernelId::blur_v_f32;
        parameters[0] = &slot->alpha_tmp_ptr_arg;
        parameters[1] = &slot->alpha_tmp_pitch_arg;
        parameters[2] = &slot->frame_width_arg;
        parameters[3] = &slot->frame_height_arg;
        parameters[4] = &slot->alpha_feather_ptr_arg;
        parameters[5] = &slot->alpha_feather_pitch_arg;
        parameters[6] = &slot->alpha_feather_radius_arg;
        break;
      case PreparedOpenCudaOperationId::background_blur_horizontal:
        prepared.request.kernel = OpenCudaVbKernelId::blur_h_u8x3;
        parameters[0] = &slot->input_ptr_arg;
        parameters[1] = &slot->input_pitch_arg;
        parameters[2] = &slot->frame_width_arg;
        parameters[3] = &slot->frame_height_arg;
        parameters[4] = &slot->blur_tmp_ptr_arg;
        parameters[5] = &slot->blur_tmp_pitch_arg;
        parameters[6] = &slot->strength_arg;
        break;
      case PreparedOpenCudaOperationId::background_blur_vertical:
        prepared.request.kernel = OpenCudaVbKernelId::blur_v_u8x3;
        parameters[0] = &slot->blur_tmp_ptr_arg;
        parameters[1] = &slot->blur_tmp_pitch_arg;
        parameters[2] = &slot->frame_width_arg;
        parameters[3] = &slot->frame_height_arg;
        parameters[4] = &slot->blurred_ptr_arg;
        parameters[5] = &slot->blurred_pitch_arg;
        parameters[6] = &slot->strength_arg;
        break;
      case PreparedOpenCudaOperationId::composite_background:
        prepared.request.kernel = OpenCudaVbKernelId::composite_background;
        parameters[0] = &slot->input_ptr_arg;
        parameters[1] = &slot->input_pitch_arg;
        parameters[2] = &slot->blurred_ptr_arg;
        parameters[3] = &slot->blurred_pitch_arg;
        parameters[4] = feather ? &slot->alpha_feather_ptr_arg
                                : &slot->resized_alpha_ptr_arg;
        parameters[5] = feather ? &slot->alpha_feather_pitch_arg
                                : &slot->resized_alpha_pitch_arg;
        parameters[6] = &slot->frame_width_arg;
        parameters[7] = &slot->frame_height_arg;
        parameters[8] = &slot->output_ptr_arg;
        parameters[9] = &slot->output_pitch_arg;
        break;
      case PreparedOpenCudaOperationId::composite_solid:
        prepared.request.kernel = OpenCudaVbKernelId::composite_solid;
        parameters[0] = &slot->input_ptr_arg;
        parameters[1] = &slot->input_pitch_arg;
        parameters[2] = feather ? &slot->alpha_feather_ptr_arg
                                : &slot->resized_alpha_ptr_arg;
        parameters[3] = feather ? &slot->alpha_feather_pitch_arg
                                : &slot->resized_alpha_pitch_arg;
        parameters[4] = &slot->frame_width_arg;
        parameters[5] = &slot->frame_height_arg;
        parameters[6] = &slot->remove_background_rgb_arg[0];
        parameters[7] = &slot->remove_background_rgb_arg[1];
        parameters[8] = &slot->remove_background_rgb_arg[2];
        parameters[9] = &slot->output_ptr_arg;
        parameters[10] = &slot->output_pitch_arg;
        break;
      case PreparedOpenCudaOperationId::composite_replace:
        prepared.request.kernel = OpenCudaVbKernelId::composite_background;
        parameters[0] = &slot->input_ptr_arg;
        parameters[1] = &slot->input_pitch_arg;
        parameters[2] = &slot->replacement_frame_ptr_arg;
        parameters[3] = &slot->replacement_frame_pitch_arg;
        parameters[4] = feather ? &slot->alpha_feather_ptr_arg
                                : &slot->resized_alpha_ptr_arg;
        parameters[5] = feather ? &slot->alpha_feather_pitch_arg
                                : &slot->resized_alpha_pitch_arg;
        parameters[6] = &slot->frame_width_arg;
        parameters[7] = &slot->frame_height_arg;
        parameters[8] = &slot->output_ptr_arg;
        parameters[9] = &slot->output_pitch_arg;
        break;
      case PreparedOpenCudaOperationId::key_light:
        prepared.request.kernel = OpenCudaVbKernelId::key_light;
        parameters[0] = &slot->input_ptr_arg;
        parameters[1] = &slot->input_pitch_arg;
        parameters[2] = &slot->resized_alpha_ptr_arg;
        parameters[3] = &slot->resized_alpha_pitch_arg;
        parameters[4] = &slot->frame_width_arg;
        parameters[5] = &slot->frame_height_arg;
        parameters[6] = &slot->key_light_r_arg;
        parameters[7] = &slot->key_light_g_arg;
        parameters[8] = &slot->key_light_b_arg;
        parameters[9] = &slot->key_light_intensity_arg;
        parameters[10] = &slot->key_light_direction_arg;
        parameters[11] = &slot->output_ptr_arg;
        parameters[12] = &slot->output_pitch_arg;
        break;
      case PreparedOpenCudaOperationId::invalid:
      case PreparedOpenCudaOperationId::upload:
      case PreparedOpenCudaOperationId::preprocess:
      case PreparedOpenCudaOperationId::inference:
      case PreparedOpenCudaOperationId::final_readback:
      case PreparedOpenCudaOperationId::completion_wait:
      case PreparedOpenCudaOperationId::final_resize:
      case PreparedOpenCudaOperationId::vignette:
        return false;
      }
      prepared.request.parameters = prepared.parameters.data();
    }
    return true;
  }

  static RuntimeStatus
  LaunchPostprocess(Slot &slot, std::size_t stage_index,
                    PreparedOpenCudaOperationId *failed_operation) noexcept {
    using cuda::kernels::PreparedOpenCudaVbKernelStatus;
    if (failed_operation)
      *failed_operation = PreparedOpenCudaOperationId::invalid;
    for (std::uint32_t index = 0;
         index < slot.postprocess_plans[stage_index].launch_count; ++index) {
      auto &prepared = slot.postprocess_launches[stage_index][index];
      PreparedOpenCudaVbKernelStatus status{};
      if (slot.postprocess.Launch(slot.postprocess_binding, prepared.request,
                                  &status)) {
        continue;
      }
      if (failed_operation)
        *failed_operation = prepared.operation;
      return status == PreparedOpenCudaVbKernelStatus::device_lost
                 ? RuntimeStatus::device_lost
                 : RuntimeStatus::postprocess_failed;
    }
    return RuntimeStatus::executed;
  }

  std::array<std::unique_ptr<Slot>, kRuntimeSlots> slots_{};
  PreparedOpenCudaRuntimeTelemetry telemetry_{};
  std::uint64_t next_handle_id_ = 1;
  cuda::CudaDriverApi *owned_driver_ = nullptr;
  cuda::CUstream owned_stream_ = nullptr;
  bool owns_stream_ = false;
  bool stream_was_owned_ = false;
  PreparedOpenCudaShutdownSnapshot shutdown_snapshot_{};
  bool shutdown_ = false;
};

} // namespace

hal::ReasonId ValidatePreparedOpenCudaProductionProgram(
    const hal::PreparedProgram &program,
    const PreparedOpenCudaProductionConfig &config) noexcept {
  return ValidateProductionProgramInternal(program, config);
}

PreparedOpenCudaVbPlan PlanPreparedOpenCudaVirtualBackground(
    const PreparedOpenCudaVbPlanInput &input) noexcept {
  return BuildPreparedOpenCudaVirtualBackgroundPlan(input);
}

struct OpenCudaPreparedResidentExecutor::Impl {
  struct Slot {
    bool active = false;
    bool quarantined = false;
    bool host_bound = false;
    bool execution_completed = false;
    hal::PreparedProgram program{};
    PreparedExecutorHandle executor_handle{};
    PreparedOpenCudaRuntimeHandle runtime_handle{};
    hal::HostFrameBinding host_binding{};
    hal::HostFrameTokens host_tokens{};
  };

  explicit Impl(Options source)
      : options(std::move(source)), model_pack(options.model_pack) {
    config.driver = options.driver;
    config.prepared_stream = options.prepared_stream;
    config.model_pack = &model_pack;
    config.model_artifact = options.model_artifact;
    config.cuda_device_id = options.cuda_device_id;
    config.adapter_instance_id = options.adapter_instance_id;
    config.stream_identity = options.stream_identity;
    config.configuration_generation = options.configuration_generation;
    config.model_preprocess_generation = options.model_preprocess_generation;
    config.model_artifact_hash = options.model_artifact_hash;
    config.executor_owns_stream = options.executor_owns_stream;
    config.strength = options.strength;
    config.remove_background_rgb = options.remove_background_rgb;
    config.replacement_rgb = options.replacement_rgb;
    config.replacement_width = options.replacement_width;
    config.replacement_height = options.replacement_height;
    config.key_light_intensity = options.key_light_intensity;
    config.key_light_temperature_preset = options.key_light_temperature_preset;
    config.key_light_direction_pan_degrees =
        options.key_light_direction_pan_degrees;
    config.vignette_intensity = options.vignette_intensity;
    config.wait_policy = options.wait_policy;
    if (options.runtime) {
      runtime = options.runtime;
    } else {
      owned_runtime = std::make_unique<ProductionPreparedOpenCudaRuntime>(
          options.driver, options.prepared_stream,
          options.executor_owns_stream);
      runtime = owned_runtime.get();
    }
  }

  Slot *Find(const PreparedExecutorHandle &handle) noexcept {
    for (auto &slot : slots) {
      if (slot.active && slot.executor_handle.handle_id == handle.handle_id &&
          slot.executor_handle.program_id == handle.program_id &&
          slot.executor_handle.execution_domain == handle.execution_domain &&
          slot.executor_handle.reserved_budget.persistent_bytes ==
              handle.reserved_budget.persistent_bytes &&
          slot.executor_handle.reserved_budget.scratch_bytes ==
              handle.reserved_budget.scratch_bytes &&
          slot.executor_handle.reserved_budget.upload_staging_bytes ==
              handle.reserved_budget.upload_staging_bytes &&
          slot.executor_handle.reserved_budget.readback_staging_bytes ==
              handle.reserved_budget.readback_staging_bytes &&
          slot.executor_handle.reserved_budget.resource_count ==
              handle.reserved_budget.resource_count &&
          slot.executor_handle.reserved_budget.in_flight_slots ==
              handle.reserved_budget.in_flight_slots &&
          slot.executor_handle.opaque == handle.opaque) {
        return &slot;
      }
    }
    return nullptr;
  }

  Options options;
  open_video::ModelPack model_pack;
  PreparedOpenCudaProductionConfig config{};
  std::unique_ptr<IPreparedOpenCudaRuntime> owned_runtime{};
  IPreparedOpenCudaRuntime *runtime = nullptr;
  std::array<Slot, kRuntimeSlots> slots{};
  ExecutorTelemetrySnapshot telemetry{};
  std::uint64_t next_handle_id = 1;
  std::uint64_t next_resource_id = 1;
  PreparedOpenCudaShutdownSnapshot shutdown_snapshot{};
  bool shutdown_requested = false;
};

OpenCudaPreparedResidentExecutor::OpenCudaPreparedResidentExecutor(
    Options options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

OpenCudaPreparedResidentExecutor::~OpenCudaPreparedResidentExecutor() {
  (void)RequestShutdown();
}

ExecutorPrepareResult OpenCudaPreparedResidentExecutor::Prepare(
    const hal::PreparedProgram &program) noexcept {
  ExecutorPrepareResult result;
  if (!impl_ || impl_->shutdown_requested || !impl_->runtime) {
    result.status = StatusId::stopped;
    result.reason = ReasonId::shutdown_requested;
    return result;
  }
  const ReasonId validation =
      ValidatePreparedOpenCudaProductionProgram(program, impl_->config);
  if (validation != ReasonId::none) {
    result.status = validation == ReasonId::resource_exhausted
                        ? StatusId::resource_exhausted
                        : StatusId::blocked;
    result.reason = validation;
    return result;
  }
  auto destination =
      std::find_if(impl_->slots.begin(), impl_->slots.end(),
                   [](const Impl::Slot &slot) { return !slot.active; });
  if (destination == impl_->slots.end()) {
    result.status = StatusId::resource_exhausted;
    result.reason = ReasonId::resource_exhausted;
    return result;
  }

  const auto prepared = impl_->runtime->Prepare(program, impl_->config);
  const auto &evidence = prepared.evidence;
  const bool model_free = program.stages[0].semantic == SemanticEffectId::vignette;
  const auto expected_provider = model_free ? kNativeProvider
      : (program.stages[program.stage_count - 1].semantic ==
                 SemanticEffectId::vignette
             ? SemanticEffectProviderId::invalid
             : kOrtProvider);
  const bool exact =
      prepared.status == RuntimeStatus::ready &&
      evidence.api_version == kPreparedOpenCudaRuntimeApiVersion &&
      evidence.status == RuntimeStatus::ready &&
      evidence.semantic == program.stages[0].semantic &&
      evidence.provider == expected_provider &&
      evidence.execution_domain == program.execution_domain &&
      evidence.model_artifact == impl_->config.model_artifact &&
      evidence.strict_cuda_only && evidence.user_stream_bound &&
      evidence.cpu_provider_fallback_disabled &&
      evidence.reusable_resources_prepared && evidence.setup_warmup_complete &&
      (model_free || evidence.inference_only_frame_path) &&
      (model_free || evidence.device_preprocess) &&
      evidence.device_postprocess && evidence.one_upload_one_readback &&
      evidence.no_intermediate_crossings && evidence.single_in_flight &&
      (model_free || ExactPreparedOpenCudaVirtualBackgroundPlan(evidence.geometry_plan)) &&
      (model_free ||
       (evidence.geometry_plan.api_version ==
            kPreparedOpenCudaRuntimeApiVersion &&
        evidence.geometry_plan.semantic == program.stages[0].semantic &&
        evidence.geometry_plan.frame_width == program.host_input.width &&
        evidence.geometry_plan.frame_height == program.host_input.height &&
        evidence.geometry_plan.model_width == static_cast<std::uint32_t>(
                                                  impl_->model_pack.matting->input.width) &&
        evidence.geometry_plan.model_height == static_cast<std::uint32_t>(
                                                   impl_->model_pack.matting->input.height) &&
        evidence.geometry_plan.strength == static_cast<std::uint32_t>(std::clamp(
                                              impl_->config.strength,
                                              video::effects::contract::kVbStrengthMin,
                                              video::effects::contract::kVbStrengthMax)))) &&
      (model_free || SameBudget(evidence.geometry_plan.required_budget,
                                evidence.reserved_budget)) &&
      evidence.reserved_budget.in_flight_slots == 1 &&
      BudgetContains(program.budget, evidence.reserved_budget) &&
      prepared.handle.handle_id != 0 &&
      prepared.handle.program_id == program.program_id &&
      prepared.handle.invalidation_key == program.invalidation_key &&
      prepared.handle.execution_domain == program.execution_domain;
  if (!exact) {
    if (prepared.handle.handle_id != 0) {
      auto rejected = prepared.handle;
      (void)impl_->runtime->DestroyPrepared(&rejected);
    }
    result.status = prepared.status == RuntimeStatus::resource_exhausted
                        ? StatusId::resource_exhausted
                        : StatusId::blocked;
    result.reason = prepared.reason == ReasonId::none
                        ? ReasonId::capability_evidence_incomplete
                        : prepared.reason;
    return result;
  }

  result.status = StatusId::ready;
  result.reason = ReasonId::none;
  result.handle.handle_id = impl_->next_handle_id++;
  if (result.handle.handle_id == 0)
    result.handle.handle_id = impl_->next_handle_id++;
  result.handle.program_id = program.program_id;
  result.handle.execution_domain = program.execution_domain;
  result.handle.reserved_budget = evidence.reserved_budget;
  const std::uint64_t output_bytes = program.host_output.planes[0].size_bytes;
  result.handle.prepared_output = MakeResource(
      program, impl_->next_resource_id++, hal::MemoryDomain::readback_staging,
      hal::ResourceLifetime::prepared_program, output_bytes,
      program.host_output.base_alignment_bytes, prepared.handle.handle_id);
  result.handle.opaque[0] = prepared.handle.handle_id;
  result.handle.opaque[1] = program.execution_domain.logical_context_generation;

  result.evidence.capability.cuda_driver_ready = true;
  result.evidence.capability.stable_device_identity = true;
  result.evidence.capability.exact_context_generation = true;
  result.evidence.capability.adapter_owned_allocator = true;
  result.evidence.capability.adapter_owned_stream = true;
  result.evidence.capability.ort_cuda_provider_ready = true;
  result.evidence.capability.ort_cuda_iobinding_ready = true;
  result.evidence.capability.ort_user_stream_bound = true;
  result.evidence.capability.explicit_ort_boundary_synchronization = false;
  result.evidence.capability.sessions_prepared = true;
  result.evidence.capability.reusable_buffers_prepared = true;
  result.evidence.capability.bounded_resources = true;
  result.evidence.capability.steady_state_allocation_free = true;
  result.evidence.capability.steady_state_probe_free = true;
  result.evidence.capability.no_cpu_fallback = true;
  result.evidence.capability.shared_frame_analysis = true;
  result.evidence.capability.completion_synchronizations_per_frame = 1;
  result.evidence.capability.device_preprocess = true;
  result.evidence.capability.device_postprocess = true;
  result.evidence.stage_count = program.stage_count;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto semantic = program.stages[index].semantic;
    result.evidence.capability.virtual_background_blur |=
        semantic == SemanticEffectId::virtual_background_blur;
    result.evidence.capability.virtual_background_remove |=
        semantic == SemanticEffectId::virtual_background_remove;
    result.evidence.capability.virtual_background_replace |=
        semantic == SemanticEffectId::virtual_background_replace;
    result.evidence.capability.virtual_key_light |=
        semantic == SemanticEffectId::virtual_key_light;
    result.evidence.capability.vignette |=
        semantic == SemanticEffectId::vignette;
    result.evidence.capability.native_vignette_kernel_prepared |=
        semantic == SemanticEffectId::vignette;
    auto &stage_evidence = result.evidence.stages[index];
    stage_evidence.stage_id = program.stages[index].stage_id;
    stage_evidence.semantic = semantic;
    stage_evidence.provider = program.stages[index].provider;
    stage_evidence.input = program.stages[index].input;
    stage_evidence.output = program.stages[index].output;
    stage_evidence.model = program.stages[index].model;
    stage_evidence.production_qualified = true;
    stage_evidence.cuda_device_input = true;
    stage_evidence.cuda_device_output = true;
    stage_evidence.cuda_iobinding = true;
    stage_evidence.device_preprocess = true;
    stage_evidence.device_postprocess = true;
    stage_evidence.shared_analysis = true;
    stage_evidence.temporal_history_device_resident = false;
    stage_evidence.cpu_fallback_disabled = true;
  }

  destination->active = true;
  destination->program = program;
  destination->executor_handle = result.handle;
  destination->runtime_handle = prepared.handle;
  return result;
}

hal::ExecuteResult OpenCudaPreparedResidentExecutor::Execute(
    const PreparedExecutorHandle &program,
    const hal::ExecuteRequest &request) noexcept {
  hal::ExecuteResult result;
  if (!impl_ || impl_->shutdown_requested || !impl_->runtime) {
    result.status = StatusId::stopped;
    result.reason = ReasonId::shutdown_requested;
    return result;
  }
  Impl::Slot *slot = impl_->Find(program);
  if (!slot) {
    result.status = StatusId::stale;
    result.reason = ReasonId::stale_program;
    return result;
  }
  if (slot->quarantined) {
    result.status = StatusId::quarantined;
    result.reason = ReasonId::device_lost;
    return result;
  }
  if (!slot->host_bound || slot->execution_completed ||
      request.capture_sequence != slot->host_binding.capture_sequence ||
      request.invalidation_key != slot->program.invalidation_key ||
      request.execution_domain != slot->program.execution_domain ||
      !SameResource(request.input, slot->host_tokens.input)) {
    result.status = StatusId::blocked;
    result.reason = !slot->host_bound ? ReasonId::host_frame_binding_invalid
                                      : ReasonId::reconfiguration_required;
    return result;
  }

  PreparedOpenCudaRuntimeFrame frame;
  frame.capture_sequence = request.capture_sequence;
  frame.input = slot->host_binding.input_planes[0].read_data;
  frame.output = slot->host_binding.output_planes[0].write_data;
  frame.input_span_bytes =
      static_cast<std::size_t>(slot->host_binding.input_planes[0].span_bytes);
  frame.output_span_bytes =
      static_cast<std::size_t>(slot->host_binding.output_planes[0].span_bytes);
  frame.input_row_stride_bytes = static_cast<std::size_t>(
      slot->host_binding.input_planes[0].row_stride_bytes);
  frame.output_row_stride_bytes = static_cast<std::size_t>(
      slot->host_binding.output_planes[0].row_stride_bytes);
  frame.stage_disabled = (request.disabled_stage_mask & 1u) != 0;
  frame.disabled_stage_mask = request.disabled_stage_mask;
  frame.stage_required = slot->program.stages[0].required;
  frame.failure_action = slot->program.stages[0].failure_action;

  const auto runtime_before = impl_->runtime->Telemetry();
  const auto executed =
      impl_->runtime->ExecuteFrame(slot->runtime_handle, frame);
  const auto runtime_after = impl_->runtime->Telemetry();
  if (!ForbiddenTelemetryUnchanged(runtime_before, runtime_after)) {
    result.status = StatusId::failed;
    result.reason = ReasonId::transfer_contract_violated;
    return result;
  }

  result.status = HalStatus(executed.status);
  result.reason = executed.reason == ReasonId::none ? HalReason(executed.status)
                                                    : executed.reason;
  result.output = slot->host_tokens.output;
  result.output_valid = executed.output_valid;
  result.stage_count = executed.stage_count;
  result.stages = executed.stages;
  if (result.stage_count == 0 && slot->program.stage_count == 1 &&
      (executed.status == RuntimeStatus::executed ||
       executed.status == RuntimeStatus::optional_failed_open ||
       executed.status == RuntimeStatus::required_failed ||
       executed.status == RuntimeStatus::device_lost ||
       executed.status == RuntimeStatus::quarantined)) {
    result.stage_count = 1;
    auto &stage = result.stages[0];
    stage.stage_id = slot->program.stages[0].stage_id;
    stage.semantic = slot->program.stages[0].semantic;
    stage.primitive = slot->program.stages[0].primitive;
    stage.elapsed_ns = executed.stage_elapsed_ns;
    if (executed.status == RuntimeStatus::executed) {
      stage.status = frame.stage_disabled
                         ? StageExecutionStatus::disabled_preserved
                         : StageExecutionStatus::succeeded;
      stage.reason = frame.stage_disabled ? ReasonId::circuit_breaker_open
                                          : ReasonId::none;
      stage.valid_output_preserved = frame.stage_disabled;
    } else if (executed.status == RuntimeStatus::optional_failed_open) {
      stage.status = StageExecutionStatus::optional_failed_open;
      stage.reason = executed.stage_failure_reason == ReasonId::none
                         ? ReasonId::optional_stage_failed_open
                         : executed.stage_failure_reason;
      stage.valid_output_preserved = executed.valid_input_preserved;
    } else {
      stage.status = StageExecutionStatus::required_failed;
      stage.reason = executed.status == RuntimeStatus::device_lost ||
                             executed.status == RuntimeStatus::quarantined
                         ? ReasonId::device_lost
                     : executed.stage_failure_reason == ReasonId::none
                         ? ReasonId::required_stage_failed
                         : executed.stage_failure_reason;
    }
  }
  result.frame_elapsed_ns = executed.frame_elapsed_ns;
  result.transfers = executed.transfers;
  if (result.stage_count == 0 && executed.status != RuntimeStatus::executed) {
    result.stage_count = 0;
  }

  impl_->telemetry.uploads += executed.transfers.uploads;
  impl_->telemetry.final_readbacks += executed.transfers.final_readbacks;
  impl_->telemetry.intermediate_readbacks +=
      executed.transfers.intermediate_readbacks;
  impl_->telemetry.intermediate_uploads +=
      executed.transfers.intermediate_uploads;
  impl_->telemetry.synchronizations += executed.transfers.synchronizations;
  impl_->telemetry.cpu_tail_boundaries +=
      executed.transfers.cpu_tail_boundaries;
  if (executed.analysis_ran)
    ++impl_->telemetry.shared_analysis_runs;

  if (executed.status == RuntimeStatus::device_lost ||
      executed.status == RuntimeStatus::quarantined) {
    slot->quarantined = true;
    result.status = StatusId::quarantined;
    result.reason = ReasonId::device_lost;
  }
  if (executed.output_valid && (result.status == StatusId::executed ||
                                result.status == StatusId::degraded)) {
    slot->execution_completed = true;
  }
  return result;
}

hal::StatusId OpenCudaPreparedResidentExecutor::BindHostFrame(
    const PreparedExecutorHandle &program, const hal::HostFrameBinding &binding,
    hal::HostFrameTokens *tokens, hal::ReasonId *reason) noexcept {
  if (reason)
    *reason = ReasonId::none;
  if (tokens)
    *tokens = {};
  if (!impl_ || impl_->shutdown_requested) {
    if (reason)
      *reason = ReasonId::shutdown_requested;
    return StatusId::stopped;
  }
  Impl::Slot *slot = impl_->Find(program);
  if (!slot) {
    if (reason)
      *reason = ReasonId::stale_program;
    return StatusId::stale;
  }
  if (!tokens || slot->quarantined || binding.input_plane_count != 1 ||
      binding.output_plane_count != 1 || binding.capture_sequence == 0 ||
      !binding.input_planes[0].read_data ||
      !binding.output_planes[0].write_data ||
      binding.input_descriptor.planes[0].row_stride_bytes !=
          binding.input_planes[0].row_stride_bytes ||
      binding.output_descriptor.planes[0].row_stride_bytes !=
          binding.output_planes[0].row_stride_bytes) {
    if (reason)
      *reason = slot->quarantined ? ReasonId::device_lost
                                  : ReasonId::host_frame_binding_invalid;
    return slot->quarantined ? StatusId::quarantined : StatusId::blocked;
  }
  if (slot->host_bound && !slot->execution_completed) {
    if (reason)
      *reason = ReasonId::output_not_committed;
    return StatusId::blocked;
  }
  slot->host_binding = binding;
  slot->host_tokens.input = MakeResource(
      slot->program, impl_->next_resource_id++,
      hal::MemoryDomain::upload_staging, hal::ResourceLifetime::frame_borrowed,
      binding.input_descriptor.planes[0].size_bytes,
      binding.input_descriptor.base_alignment_bytes,
      reinterpret_cast<std::uintptr_t>(binding.input_planes[0].read_data));
  slot->host_tokens.output = MakeResource(
      slot->program, impl_->next_resource_id++,
      hal::MemoryDomain::readback_staging,
      hal::ResourceLifetime::frame_borrowed,
      binding.output_descriptor.planes[0].size_bytes,
      binding.output_descriptor.base_alignment_bytes,
      reinterpret_cast<std::uintptr_t>(binding.output_planes[0].write_data));
  slot->host_bound = true;
  slot->execution_completed = false;
  *tokens = slot->host_tokens;
  return StatusId::ready;
}

hal::StatusId OpenCudaPreparedResidentExecutor::ResolveHostFrameOutput(
    const PreparedExecutorHandle &program,
    const hal::CompletionToken &completion, const hal::ResourceHandle &output,
    hal::HostFrameOutputResolution *resolution,
    hal::ReasonId *reason) noexcept {
  if (reason)
    *reason = ReasonId::none;
  if (resolution)
    *resolution = {};
  if (!impl_ || impl_->shutdown_requested) {
    if (reason)
      *reason = ReasonId::shutdown_requested;
    return StatusId::stopped;
  }
  Impl::Slot *slot = impl_->Find(program);
  if (!slot) {
    if (reason)
      *reason = ReasonId::stale_program;
    return StatusId::stale;
  }
  if (!resolution || !slot->host_bound || !slot->execution_completed ||
      !completion.complete ||
      completion.program_id != slot->program.program_id ||
      completion.capture_sequence != slot->host_binding.capture_sequence ||
      !SameResource(output, slot->host_tokens.output)) {
    if (reason)
      *reason = ReasonId::output_not_committed;
    return StatusId::blocked;
  }
  resolution->descriptor = slot->host_binding.output_descriptor;
  resolution->planes = slot->host_binding.output_planes;
  resolution->plane_count = slot->host_binding.output_plane_count;
  resolution->capture_sequence = slot->host_binding.capture_sequence;
  resolution->output_ready = true;
  return StatusId::ready;
}

hal::StatusId OpenCudaPreparedResidentExecutor::ResetTemporalState(
    const PreparedExecutorHandle &program, hal::ReasonId reason) noexcept {
  if (!impl_ || impl_->shutdown_requested || !impl_->runtime)
    return StatusId::stopped;
  Impl::Slot *slot = impl_->Find(program);
  if (!slot)
    return StatusId::stale;
  const auto status =
      impl_->runtime->ResetTemporalState(slot->runtime_handle, reason);
  if (status == RuntimeStatus::device_lost ||
      status == RuntimeStatus::quarantined)
    slot->quarantined = true;
  return HalStatus(status);
}

void OpenCudaPreparedResidentExecutor::DestroyPrepared(
    PreparedExecutorHandle *program) noexcept {
  if (!program || !impl_)
    return;
  Impl::Slot *slot = impl_->Find(*program);
  if (slot && impl_->runtime) {
    (void)impl_->runtime->DestroyPrepared(&slot->runtime_handle);
    *slot = {};
  }
  *program = {};
}

PreparedOpenCudaShutdownSnapshot
OpenCudaPreparedResidentExecutor::RequestShutdown() noexcept {
  PreparedOpenCudaShutdownSnapshot result;
  result.status = RuntimeStatus::ready;
  result.shutdown_requested = true;
  result.quiescent = true;
  result.driver_context_release_permitted = true;
  if (!impl_)
    return result;

  if (!impl_->shutdown_requested) {
    impl_->shutdown_requested = true;
    for (auto &slot : impl_->slots) {
      if (slot.active && impl_->runtime)
        (void)impl_->runtime->DestroyPrepared(&slot.runtime_handle);
      slot = {};
    }
  }
  if (impl_->runtime)
    impl_->shutdown_snapshot = impl_->runtime->RequestShutdown();
  else
    impl_->shutdown_snapshot = result;
  return impl_->shutdown_snapshot;
}

PreparedOpenCudaShutdownSnapshot
OpenCudaPreparedResidentExecutor::ShutdownSnapshot() const noexcept {
  PreparedOpenCudaShutdownSnapshot result;
  result.status = RuntimeStatus::ready;
  result.shutdown_requested = true;
  result.quiescent = true;
  result.driver_context_release_permitted = true;
  if (!impl_)
    return result;
  if (impl_->runtime && impl_->shutdown_requested)
    return impl_->runtime->ShutdownSnapshot();
  return impl_->shutdown_snapshot;
}

void OpenCudaPreparedResidentExecutor::Shutdown() noexcept {
  (void)RequestShutdown();
}

ExecutorTelemetrySnapshot
OpenCudaPreparedResidentExecutor::Telemetry() const noexcept {
  if (!impl_)
    return {};
  ExecutorTelemetrySnapshot result = impl_->telemetry;
  if (impl_->runtime) {
    const auto runtime = impl_->runtime->Telemetry();
    result.allocation_requests = runtime.allocation_requests;
    result.provider_probes = runtime.provider_probes;
    result.model_scans = runtime.model_scans;
    result.manifest_parses = runtime.manifest_parses;
    result.session_constructions = runtime.session_constructions;
    result.json_serializations = runtime.json_serializations;
  }
  return result;
}

} // namespace studiocast::open_cuda
