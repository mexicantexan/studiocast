#include "core/vulkan/qualified_final_resident_executor.h"

#include <algorithm>
#include <chrono>
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
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    const auto semantic = program.stages[i].semantic;
    if (semantic == SemanticEffectId::vignette) {
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
         mirror == options.request_mirror;
}

} // namespace

QualifiedFinalResidentExecutor::QualifiedFinalResidentExecutor(
    std::uint64_t adapter_instance_id,
    kernels::UtilityKernels &kernels) noexcept
    : adapter_instance_id_(adapter_instance_id), kernels_(&kernels) {
  RefreshEvidence();
}

QualifiedFinalResidentExecutor::~QualifiedFinalResidentExecutor() {
  Shutdown();
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
  evidence_.mirror_production_ready = common;
  evidence_.vignette_fixed_center_production_ready = common;
  evidence_.blocker =
      common ? ReasonId::none : ReasonId::capability_evidence_incomplete;
}

VulkanResidentCapabilityEvidence
QualifiedFinalResidentExecutor::CapabilityEvidence() const noexcept {
  return evidence_;
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
           VulkanVignetteCenterMode::fixed_frame_center) ||
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
    budget_valid =
        budget_valid &&
        Add(&used.persistent_bytes, slot->mirror_output.byte_size());
    ++used.resource_count;
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
    result.reason = ReasonId::device_lost;
    result.device_lost = true;
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
      (options.request_vignette &&
       (options.vignette_center != slot->options.vignette_center ||
        options.vignette_intensity_percent !=
            slot->options.vignette_intensity_percent)) ||
      !MatchesStageSubset(program, options) || !binding_valid) {
    result.status = StatusId::blocked;
    result.reason = ReasonId::malformed_contract;
    return finish();
  }
  const std::uint64_t submissions_before =
      kernels_->synchronous_submission_count();
  if (!kernels_->UploadU8x3ToDeviceLocal(
          binding.input, binding.input_row_stride_bytes, slot->upload_staging,
          slot->source, nullptr)) {
    const auto failed_health = kernels_->device()->health_state();
    result.device_lost = failed_health.health != VulkanContextHealth::healthy;
    slot->quarantined = result.device_lost;
    result.status =
        result.device_lost ? StatusId::quarantined : StatusId::failed;
    result.reason = result.device_lost ? ReasonId::device_lost
                                       : ReasonId::synchronization_failed;
    return finish();
  }
  result.transfers.uploads = 1;

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
  input.source = &slot->source;
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
    result.device_lost = failed_health.health != VulkanContextHealth::healthy;
    slot->quarantined = result.device_lost;
    result.status =
        result.device_lost ? StatusId::quarantined : StatusId::failed;
    result.reason = result.device_lost ? ReasonId::device_lost
                                       : ReasonId::output_write_failed;
    result.transfers.synchronizations =
        kernels_->synchronous_submission_count() - submissions_before;
    return finish();
  }
  result.transfers.final_readbacks = 1;
  result.transfers.synchronizations =
      kernels_->synchronous_submission_count() - submissions_before;
  result.output_written = true;
  // Transport completion is separate from per-stage policy. The common
  // adapter derives required failure or optional degradation while retaining
  // this successfully written output.
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

void QualifiedFinalResidentExecutor::Release(Slot *slot) noexcept {
  if (!slot)
    return;
  slot->vignette.Shutdown();
  slot->readback_staging.Free();
  slot->mirror_output.Free();
  slot->vignette_output.Free();
  slot->resize_scratch.Free();
  slot->source.Free();
  slot->upload_staging.Free();
  slot->occupied = false;
  slot->quarantined = false;
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
  if (Slot *slot = Find(prepared->state_id))
    Release(slot);
  *prepared = {};
}

void QualifiedFinalResidentExecutor::Shutdown() noexcept {
  if (shutdown_)
    return;
  for (auto &slot : slots_)
    Release(&slot);
  shutdown_ = true;
}

} // namespace studiocast::vulkan
