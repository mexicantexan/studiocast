#include "core/open_video/production_video_backend_setup_source.h"

#include "core/open_video/model_pack_registry.h"
#include "core/open_video/video_execution_factory.h"
#include "core/video/effects/broadcast_effect_contract.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

namespace studiocast::open_cuda {
namespace {

constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
constexpr std::uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kHashPrime = UINT64_C(1099511628211);

bool IsZeroHash(const std::array<std::uint8_t, 32> &hash) noexcept {
  for (const auto value : hash) {
    if (value != 0)
      return false;
  }
  return true;
}

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

video::ProductionVideoResidentSetupSourceResult
Failure(hal::StatusId status, hal::ReasonId reason) noexcept {
  video::ProductionVideoResidentSetupSourceResult result;
  result.status = status;
  result.reason = reason;
  result.readiness.status = status;
  result.readiness.reason = reason;
  result.readiness.requested_backend = hal::BackendId::cuda;
  result.readiness.selected_backend = hal::BackendId::invalid;
  result.readiness.provider = kProvider;
  result.setup_completed_outside_frame_loop = true;
  return result;
}

std::uint64_t HashValue(std::uint64_t hash, std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    hash ^= static_cast<std::uint8_t>(value >> shift);
    hash *= kHashPrime;
  }
  return hash;
}

template <typename Range>
std::uint64_t HashBytes(std::uint64_t hash, const Range &bytes) noexcept {
  for (const auto value : bytes) {
    hash ^= static_cast<std::uint8_t>(value);
    hash *= kHashPrime;
  }
  return hash;
}

std::uint64_t HashArtifact(
    const open_video::ValidatedModelArtifactIdentity &identity) noexcept {
  std::uint64_t hash = kHashOffset;
  hash = HashValue(hash, identity.schema_version);
  hash = HashValue(hash, identity.registry_generation);
  hash = HashBytes(hash, identity.model_id_sha256);
  hash = HashBytes(hash, identity.manifest_sha256);
  hash = HashBytes(hash, identity.artifact_sha256);
  hash = HashBytes(hash, identity.artifact_set_sha256);
  return hash == 0 ? 1 : hash;
}

const open_video::ModelFile *
MainOnnx(const open_video::ModelPack &pack) noexcept {
  for (const auto &file : pack.files) {
    if (file.kind == "onnx" && (file.role == "main" || file.role.empty()))
      return &file;
  }
  return nullptr;
}

bool Add(std::uint64_t value, std::uint64_t *sum) noexcept {
  if (sum == nullptr ||
      value > std::numeric_limits<std::uint64_t>::max() - *sum) {
    return false;
  }
  *sum += value;
  return true;
}

bool Multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t *product) noexcept {
  if (product == nullptr ||
      (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *product = left * right;
  return true;
}

bool AlignRow(std::uint64_t active, std::uint64_t *aligned) noexcept {
  constexpr std::uint64_t kAlignment = 4096;
  if (aligned == nullptr ||
      active > std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1)) {
    return false;
  }
  *aligned = ((active + kAlignment - 1) / kAlignment) * kAlignment;
  return true;
}

bool ResourceBudgetFor(std::uint32_t frame_width, std::uint32_t frame_height,
                       std::uint32_t model_width, std::uint32_t model_height,
                       hal::SemanticEffectId semantic, int strength,
                       hal::ResourceBudget *budget) noexcept {
  if (budget == nullptr || frame_width == 0 || frame_height == 0 ||
      model_width == 0 || model_height == 0) {
    return false;
  }
  std::uint64_t rgb_row = 0;
  std::uint64_t alpha_row = 0;
  std::uint64_t frame_rgb = 0;
  std::uint64_t frame_alpha = 0;
  std::uint64_t model_alpha_row = 0;
  std::uint64_t model_tensor_row = 0;
  std::uint64_t model_alpha = 0;
  std::uint64_t model_tensor = 0;
  std::uint64_t boundary = 0;
  if (!AlignRow(static_cast<std::uint64_t>(frame_width) * 3u, &rgb_row) ||
      !AlignRow(static_cast<std::uint64_t>(frame_width) * 4u, &alpha_row) ||
      !Multiply(rgb_row, frame_height, &frame_rgb) ||
      !Multiply(alpha_row, frame_height, &frame_alpha) ||
      !Multiply(model_width, 4u, &model_alpha_row) ||
      !AlignRow(model_alpha_row, &model_alpha_row) ||
      !Multiply(model_alpha_row, model_height, &model_alpha) ||
      !Multiply(model_width, 12u, &model_tensor_row) ||
      !AlignRow(model_tensor_row, &model_tensor_row) ||
      !Multiply(model_tensor_row, model_height, &model_tensor) ||
      !Multiply(frame_width, frame_height, &boundary) ||
      !Multiply(boundary, 3u, &boundary)) {
    return false;
  }

  hal::ResourceBudget result;
  if (!Add(frame_rgb, &result.persistent_bytes) ||
      !Add(frame_rgb, &result.persistent_bytes) ||
      !Add(model_alpha, &result.persistent_bytes) ||
      !Add(model_tensor, &result.persistent_bytes) ||
      !Add(frame_alpha, &result.persistent_bytes)) {
    return false;
  }
  const int canonical_strength =
      std::clamp(strength, video::effects::contract::kVbStrengthMin,
                 video::effects::contract::kVbStrengthMax);
  const bool feather = std::min(4, canonical_strength / 16) > 0;
  const bool blur = semantic == hal::SemanticEffectId::virtual_background_blur;
  if (feather && (!Add(frame_alpha, &result.scratch_bytes) ||
                  !Add(frame_alpha, &result.scratch_bytes))) {
    return false;
  }
  if (blur && (!Add(frame_rgb, &result.scratch_bytes) ||
               !Add(frame_rgb, &result.scratch_bytes))) {
    return false;
  }
  result.scratch_bytes = std::max<std::uint64_t>(result.scratch_bytes, 1);
  result.upload_staging_bytes = boundary;
  result.readback_staging_bytes = boundary;
  result.resource_count = (blur ? 19u : 17u) + (feather ? 2u : 0u);
  result.in_flight_slots = 1;
  if (result.persistent_bytes > (UINT64_C(2) << 30) ||
      result.scratch_bytes > (UINT64_C(1) << 30) ||
      result.upload_staging_bytes > (UINT64_C(1) << 30) ||
      result.readback_staging_bytes > (UINT64_C(1) << 30)) {
    return false;
  }
  *budget = result;
  return true;
}

bool TightRgb24(const hal::ImageDescriptor &image,
                hal::Mutability mutability) noexcept {
  if (image.width == 0 || image.height == 0 || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != hal::HostCoherence::coherent ||
      image.mutability != mutability ||
      image.aliasing != hal::AliasingPermission::forbidden ||
      image.in_place_permitted || image.base_alignment_bytes != 1 ||
      image.planes[0].offset_bytes != 0 ||
      image.planes[0].alignment_bytes != 1 ||
      image.planes[0].width != image.width ||
      image.planes[0].height != image.height) {
    return false;
  }
  const auto stride = static_cast<std::uint64_t>(image.width) * 3u;
  return image.planes[0].row_stride_bytes == stride &&
         image.planes[0].size_bytes ==
             stride * static_cast<std::uint64_t>(image.height);
}

int Strength(const video::CanonicalVideoStage &stage) noexcept {
  for (std::size_t index = 0; index < stage.parameter_count; ++index) {
    const auto &parameter = stage.parameters[index];
    if (parameter.id == video::CanonicalVideoParameterId::strength &&
        parameter.kind == video::CanonicalVideoParameterKind::signed_integer) {
      return static_cast<int>(parameter.signed_value);
    }
  }
  return video::effects::contract::kVbStrengthDefault;
}

std::array<std::uint8_t, 3>
RemoveColor(const video::CanonicalVideoStage &stage) noexcept {
  for (std::size_t index = 0; index < stage.parameter_count; ++index) {
    const auto &parameter = stage.parameters[index];
    if (parameter.id == video::CanonicalVideoParameterId::remove_color_rgb &&
        parameter.kind ==
            video::CanonicalVideoParameterKind::unsigned_integer) {
      return {
          static_cast<std::uint8_t>((parameter.unsigned_value >> 16u) & 0xffu),
          static_cast<std::uint8_t>((parameter.unsigned_value >> 8u) & 0xffu),
          static_cast<std::uint8_t>(parameter.unsigned_value & 0xffu),
      };
    }
  }
  return {0, 0, 0};
}

bool ResolveModel(const video::CanonicalVideoStage &stage,
                  const std::string &requested_model_id,
                  open_video::ModelPack *pack,
                  open_video::ValidatedModelArtifactIdentity *identity) {
  if (pack == nullptr || identity == nullptr)
    return false;
  const auto registry = open_video::ModelPackRegistry::ScanDefault();
  const bool model_unbound = IsZeroHash(stage.model.model_id_hash);
  const bool artifact_unbound = IsZeroHash(stage.model.artifact_set_hash);
  if (model_unbound != artifact_unbound)
    return false;

  const auto accept = [&](const open_video::ModelPack &candidate) {
    if (candidate.task != "matting" || !candidate.matting)
      return false;
    const auto *artifact = MainOnnx(candidate);
    if (artifact == nullptr)
      return false;
    const auto validation =
        open_video::ValidateModelArtifactIdentity(candidate, *artifact);
    if (!validation.ok())
      return false;
    if (!model_unbound &&
        (stage.model.model_id_hash != validation.identity.model_id_sha256 ||
         stage.model.artifact_set_hash !=
             validation.identity.artifact_set_sha256 ||
         (stage.model.artifact_generation != 0 &&
          stage.model.artifact_generation !=
              validation.identity.registry_generation))) {
      return false;
    }
    *pack = candidate;
    *identity = validation.identity;
    return true;
  };

  if (!requested_model_id.empty()) {
    const auto requested = registry.Find("matting", requested_model_id);
    return requested.has_value() && accept(*requested);
  }
  for (const auto &candidate : registry.ListModels()) {
    if (accept(candidate))
      return true;
  }
  return false;
}

class ProductionSetupSource final
    : public video::IProductionVideoBackendSetupSource {
public:
  video::ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    const auto &caller = request.plan;
    if (request.contract_version !=
            video::kProductionVideoBackendManagerContractVersion ||
        caller.intent.backend.mode !=
            hal::BackendRequestMode::explicit_backend ||
        caller.intent.backend.backend != hal::BackendId::cuda ||
        !caller.intent.backend.strict ||
        caller.intent.preferences.compute_family !=
            hal::ComputeFamilyAllowance::cuda_family ||
        caller.intent.preferences.effects_engine !=
            hal::EffectsEnginePreferenceConstraint::open_cuda ||
        !caller.intent.preferences.strict_effects_engine ||
        caller.intent.requested_provider != kProvider ||
        !caller.intent.strict_provider_selection) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::strict_backend_unavailable);
    }
    if (caller.intent.stage_count != 1) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::cross_adapter_residency_forbidden);
    }
    const auto &stage = caller.intent.stages[0];
    if (stage.intent.semantic !=
            hal::SemanticEffectId::virtual_background_blur &&
        stage.intent.semantic !=
            hal::SemanticEffectId::virtual_background_remove) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::semantic_effect_unsupported);
    }
    if (!TightRgb24(caller.host_input, hal::Mutability::read_only) ||
        !TightRgb24(caller.host_output, hal::Mutability::write_only) ||
        caller.host_input.width != caller.host_output.width ||
        caller.host_input.height != caller.host_output.height) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::malformed_descriptor);
    }
    if (caller.cpu_tail_count != 1 ||
        caller.cpu_tails[0].id != hal::CpuTailId::writer_copy ||
        caller.cpu_tails[0].after_stage_id != stage.intent.stage_id ||
        caller.cpu_tails[0].placement !=
            hal::DeclaredCpuTail::Placement::after_stage) {
      return Failure(hal::StatusId::blocked, hal::ReasonId::cpu_tail_required);
    }

    try {
      open_video::ModelPack pack;
      open_video::ValidatedModelArtifactIdentity artifact;
      if (!ResolveModel(stage, request.effects.virtual_background.model_id,
                        &pack, &artifact)) {
        return Failure(hal::StatusId::unavailable,
                       hal::ReasonId::capability_evidence_incomplete);
      }
      const auto &matting = *pack.matting;
      hal::ResourceBudget budget;
      const int strength = Strength(stage);
      if (!ResourceBudgetFor(caller.host_input.width, caller.host_input.height,
                             static_cast<std::uint32_t>(matting.input.width),
                             static_cast<std::uint32_t>(matting.input.height),
                             stage.intent.semantic, strength, &budget)) {
        return Failure(hal::StatusId::resource_exhausted,
                       hal::ReasonId::resource_exhausted);
      }

      static std::atomic<std::uint64_t> next_adapter{
          UINT64_C(0x4f43424d00000000)};
      std::uint64_t adapter_instance_id =
          next_adapter.fetch_add(1, std::memory_order_relaxed);
      if (adapter_instance_id == 0) {
        adapter_instance_id =
            next_adapter.fetch_add(1, std::memory_order_relaxed);
      }

      OpenCudaProductionVideoExecutionFactoryRequest factory;
      factory.factory_id = adapter_instance_id ^ UINT64_C(0x4f43534554555031);
      if (factory.factory_id == 0)
        factory.factory_id = 1;
      factory.selection_priority = 100;
      factory.semantic = stage.intent.semantic;
      factory.model.model_id_hash = artifact.model_id_sha256;
      factory.model.artifact_set_hash = artifact.artifact_set_sha256;
      factory.model.artifact_generation = artifact.registry_generation;
      factory.generations = caller.invalidation_key.generations;
      factory.generations.model = artifact.registry_generation;
      if (factory.generations.provider == 0)
        factory.generations.provider = 1;
      if (factory.generations.device == 0)
        factory.generations.device = 1;
      factory.reserved_budget = budget;
      factory.backend.adapter_instance_id = adapter_instance_id;
      factory.backend.cuda_device_id = 0;
      factory.backend.model_pack = pack;
      factory.backend.model_artifact = artifact;
      factory.backend.configuration_generation =
          factory.generations.configuration;
      factory.backend.model_preprocess_generation = factory.generations.model;
      factory.backend.model_artifact_hash = HashArtifact(artifact);
      factory.backend.strength = strength;
      factory.backend.remove_background_rgb = RemoveColor(stage);
      factory.backend.wait_policy.timeout_nanoseconds =
          request.wait.timeout_nanoseconds;
      factory.backend.wait_policy.poll_interval_nanoseconds =
          request.wait.poll_interval_nanoseconds;
      factory.backend.wait_policy.maximum_poll_count =
          request.wait.maximum_poll_count;
      factory.backend.wait_policy.stop_requested = request.wait.stop_requested;
      factory.backend.wait_policy.cancel_requested =
          request.wait.cancel_requested;
      factory.backend.wait_policy.now_nanoseconds =
          request.wait.now_nanoseconds;
      factory.backend.wait_policy.poll_wait = request.wait.poll_wait;
      factory.backend.wait_policy.callback_context =
          request.wait.callback_context;
      factory.model_artifact_validated = true;
      factory.provider_runtime_validated = true;
      factory.cuda_iobinding_validated = true;
      factory.user_stream_bound = true;
      factory.reusable_resources_supported = true;
      factory.device_preprocess_supported = true;
      factory.device_postprocess_supported = true;
      factory.stop_aware_bounded_wait = true;
      factory.allocation_free_execute = true;

      video::ProductionVideoResidentSetupSourceResult result;
      result.execution_kind = video::ProductionVideoResidentSetupSourceResult::
          ExecutionKind::resident_island;
      result.canonical_effects = request.effects;
      result.setup_completed_outside_frame_loop = true;
      OpenCudaProductionVideoExecutionBootstrapResult bootstrap;
      hal::ReasonId reason = hal::ReasonId::none;
      result.status =
          OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
              factory, &result.lease, &bootstrap, &result.lease_acquire,
              &reason);
      result.reason = reason;
      result.readiness = bootstrap.setup_lease.readiness;
      if (!IsSetupSuccess(result.status) || !result.lease.live())
        return result;

      result.plan = caller;
      result.plan.intent.stages[0].model =
          bootstrap.setup_lease.artifacts[0].artifact;
      result.plan.invalidation_key.generations =
          bootstrap.setup_lease.generations;
      result.plan.invalidation_key.model_artifact_hash = HashArtifact(artifact);
      result.plan.invalidation_key.execution_domain_hash =
          video::ProductionVideoExecutionDomainHash(
              bootstrap.setup_lease.execution_domain);
      result.plan.budget = bootstrap.setup_lease.reserved_budget;
      auto &selection = result.model_selections[result.model_selection_count++];
      selection.semantic = stage.intent.semantic;
      selection.requested_model_id =
          request.effects.virtual_background.model_id;
      selection.selected_model_id = pack.id;
      selection.artifact = bootstrap.setup_lease.artifacts[0].artifact;
      selection.default_selected =
          request.effects.virtual_background.model_id.empty();
      return result;
    } catch (const std::bad_alloc &) {
      return Failure(hal::StatusId::resource_exhausted,
                     hal::ReasonId::resource_exhausted);
    } catch (...) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::capability_evidence_incomplete);
    }
  }
};

} // namespace

std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSource() noexcept {
  try {
    return std::make_unique<ProductionSetupSource>();
  } catch (...) {
    return nullptr;
  }
}

} // namespace studiocast::open_cuda
