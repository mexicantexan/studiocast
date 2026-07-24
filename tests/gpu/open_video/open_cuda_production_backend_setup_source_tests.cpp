#include "core/open_video/production_video_backend_setup_source.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

using namespace studiocast;

bool Check(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << "\n";
  return condition;
}

hal::ImageDescriptor HostRgb(std::uint32_t width, std::uint32_t height,
                             hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = width;
  image.height = height;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.alpha_meaning = hal::AlphaMeaning::none;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::coherent;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = static_cast<std::uint64_t>(width) * 3u;
  image.planes[0].size_bytes = image.planes[0].row_stride_bytes * height;
  image.planes[0].width = width;
  image.planes[0].height = height;
  image.planes[0].alignment_bytes = 1;
  return image;
}

video::ProductionVideoResidentSetup
StrictRequest(hal::SemanticEffectId semantic =
                  hal::SemanticEffectId::virtual_background_blur) {
  video::ProductionVideoResidentSetup request;
  auto &plan = request.plan;
  plan.plan_id = 51;
  plan.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  plan.intent.backend.backend = hal::BackendId::cuda;
  plan.intent.backend.strict = true;
  plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::open_cuda;
  plan.intent.preferences.strict_effects_engine = true;
  plan.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  plan.intent.strict_provider_selection = true;
  plan.intent.stage_count = 1;
  auto &stage = plan.intent.stages[0];
  stage.intent.stage_id = 7;
  stage.intent.semantic = semantic;
  stage.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
  stage.parameters[0] = {
      video::CanonicalVideoParameterId::strength,
      video::CanonicalVideoParameterKind::signed_integer,
      8,
  };
  stage.parameter_count = 1;
  plan.host_input = HostRgb(64, 48, hal::Mutability::read_only);
  plan.host_output = HostRgb(64, 48, hal::Mutability::write_only);
  plan.invalidation_key.generations.configuration = 1;
  plan.invalidation_key.generations.geometry = 1;
  plan.invalidation_key.canonical_plan_hash = 1;
  plan.budget.persistent_bytes = UINT64_C(512) << 20;
  plan.budget.scratch_bytes = UINT64_C(256) << 20;
  plan.budget.upload_staging_bytes = UINT64_C(64) << 20;
  plan.budget.readback_staging_bytes = UINT64_C(64) << 20;
  plan.budget.resource_count = 128;
  plan.budget.in_flight_slots = 1;
  plan.cpu_tails[0].id = hal::CpuTailId::writer_copy;
  plan.cpu_tails[0].after_stage_id = stage.intent.stage_id;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].boundary_index = 1;
  plan.cpu_tail_count = 1;
  plan.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  plan.migration_policy = video::VideoExecutionMigrationPolicy::prepared_only;
  return request;
}

bool StrictPolicyIsFailClosed() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  if (!Check(source != nullptr, "source construction failed"))
    return false;

  auto request = StrictRequest();
  request.plan.intent.backend.strict = false;
  auto result = source->AcquireResidentSetup(request);
  bool ok =
      Check(result.status == hal::StatusId::blocked &&
                result.reason == hal::ReasonId::strict_backend_unavailable &&
                result.setup_completed_outside_frame_loop,
            "non-strict backend request did not fail closed");

  request = StrictRequest();
  request.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  result = source->AcquireResidentSetup(request);
  ok &= Check(result.status == hal::StatusId::blocked &&
                  result.reason == hal::ReasonId::strict_backend_unavailable,
              "CPU provider was accepted by strict CUDA source");

  request = StrictRequest();
  request.plan.intent.stage_count = 2;
  result = source->AcquireResidentSetup(request);
  ok &= Check(result.status == hal::StatusId::blocked &&
                  result.reason ==
                      hal::ReasonId::cross_adapter_residency_forbidden,
              "mixed resident section was not rejected");
  return ok;
}

bool SemanticCapabilityClaimsRemainExact() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  const auto temp = std::filesystem::temp_directory_path() /
                    "studiocast-open-cuda-capability-empty";
  std::error_code ec;
  std::filesystem::remove_all(temp, ec);
  std::filesystem::create_directories(temp, ec);
  const char *old = std::getenv("XDG_DATA_HOME");
  const std::string old_value = old == nullptr ? std::string{} : old;
  const bool had_old = old != nullptr;
  (void)setenv("XDG_DATA_HOME", temp.c_str(), 1);
  const hal::SemanticEffectId supported[] = {
      hal::SemanticEffectId::virtual_background_replace,
      hal::SemanticEffectId::virtual_key_light,
  };
  bool ok = true;
  for (const auto semantic : supported) {
    const auto result = source->AcquireResidentSetup(StrictRequest(semantic));
    ok &= Check(
        result.status == hal::StatusId::unavailable &&
            result.reason == hal::ReasonId::capability_evidence_incomplete &&
            result.setup_completed_outside_frame_loop && !result.lease.live(),
        "supported semantic did not advance to setup-time model evidence");
  }
  if (had_old)
    (void)setenv("XDG_DATA_HOME", old_value.c_str(), 1);
  else
    (void)unsetenv("XDG_DATA_HOME");
  std::filesystem::remove_all(temp, ec);

  const hal::SemanticEffectId unsupported[] = {
      hal::SemanticEffectId::auto_frame,
      hal::SemanticEffectId::mirror,
      hal::SemanticEffectId::vignette,
  };
  for (const auto semantic : unsupported) {
    const auto result = source->AcquireResidentSetup(StrictRequest(semantic));
    ok &=
        Check(result.status == hal::StatusId::blocked &&
                  result.reason == hal::ReasonId::semantic_effect_unsupported &&
                  result.setup_completed_outside_frame_loop,
              "unsupported semantic gained false CUDA capability");
  }
  const hal::SemanticEffectId cpu_provider_compatibility[] = {
      hal::SemanticEffectId::video_noise_removal,
      hal::SemanticEffectId::eye_contact,
  };
  for (const auto semantic : cpu_provider_compatibility) {
    const auto result = source->AcquireResidentSetup(StrictRequest(semantic));
    ok &= Check(
        result.status == hal::StatusId::blocked &&
            result.reason == hal::ReasonId::semantic_provider_unavailable &&
            result.setup_completed_outside_frame_loop && !result.lease.live() &&
            !result.compatibility_executor,
        "strict CUDA accepted a CPU-provider compatibility semantic");
  }
  return ok;
}

bool HostAndModelEvidenceAreSetupOnly() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  auto request = StrictRequest();
  request.plan.host_output.width += 1;
  auto result = source->AcquireResidentSetup(request);
  bool ok = Check(result.status == hal::StatusId::blocked &&
                      result.reason == hal::ReasonId::malformed_descriptor,
                  "geometry mismatch was not rejected before setup");

  const auto temp = std::filesystem::temp_directory_path() /
                    "studiocast-open-cuda-backend-source-empty";
  std::error_code ec;
  std::filesystem::create_directories(temp, ec);
  const char *old = std::getenv("XDG_DATA_HOME");
  const std::string old_value = old == nullptr ? std::string{} : old;
  const bool had_old = old != nullptr;
  (void)setenv("XDG_DATA_HOME", temp.c_str(), 1);
  result = source->AcquireResidentSetup(StrictRequest());
  if (had_old)
    (void)setenv("XDG_DATA_HOME", old_value.c_str(), 1);
  else
    (void)unsetenv("XDG_DATA_HOME");
  std::filesystem::remove_all(temp, ec);

  ok &= Check(
      result.status == hal::StatusId::unavailable &&
          result.reason == hal::ReasonId::capability_evidence_incomplete &&
          result.setup_completed_outside_frame_loop && !result.lease.live(),
      "missing model evidence did not fail during setup");
  return ok;
}

bool CompatibilityTopologyNeverMasqueradesAsStrictCuda() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  auto request = StrictRequest(hal::SemanticEffectId::video_noise_removal);
  request.plan.intent.stage_count = 2;
  request.plan.intent.stages[1].intent.stage_id = 8;
  request.plan.intent.stages[1].intent.semantic =
      hal::SemanticEffectId::eye_contact;
  request.plan.intent.stages[1].intent.failure_action =
      hal::FailureAction::fail_open_preserve_input;
  auto result = source->AcquireResidentSetup(request);
  bool ok =
      Check(result.status == hal::StatusId::blocked &&
                result.reason == hal::ReasonId::semantic_provider_unavailable &&
                !result.lease.live() && !result.compatibility_executor,
            "strict CUDA accepted CPU-tensor compatibility providers");

  request.plan.intent.backend.mode = hal::BackendRequestMode::automatic;
  request.plan.intent.backend.backend = hal::BackendId::invalid;
  request.plan.intent.backend.strict = false;
  request.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::any;
  request.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  request.plan.intent.preferences.strict_effects_engine = false;
  request.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::invalid;
  request.plan.intent.strict_provider_selection = false;
  request.effects.video_noise_removal.enabled = true;
  request.effects.eye_contact.enabled = true;
  result = source->AcquireResidentSetup(request);
  ok &= Check(
      result.status == hal::StatusId::unavailable &&
          result.reason == hal::ReasonId::capability_evidence_incomplete &&
          result.setup_completed_outside_frame_loop && !result.lease.live(),
      "automatic compatibility did not fail during model setup");
  return ok;
}

} // namespace

int main() {
  if (!StrictPolicyIsFailClosed() || !SemanticCapabilityClaimsRemainExact() ||
      !HostAndModelEvidenceAreSetupOnly() ||
      !CompatibilityTopologyNeverMasqueradesAsStrictCuda()) {
    return 1;
  }
  std::cout << "Open CUDA production backend setup-source tests passed.\n";
  return 0;
}
