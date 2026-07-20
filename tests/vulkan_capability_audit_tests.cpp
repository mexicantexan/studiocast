#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

#include "core/video/effects/broadcast_effect_contract.h"

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR ""
#endif

namespace {

namespace fs = std::filesystem;
using namespace studiocast::video::effects::contract;

struct CapabilityRow {
  std::string_view effect_id;
  std::string_view status;
  std::string_view reason_code;
};

constexpr std::array<CapabilityRow, 9> kExpectedCapabilities{{
    {kEffectIdMirror, "production usable",
     "open_vulkan_mirror_production_ready"},
    {kEffectIdVirtualBackgroundBlur, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdVirtualBackgroundRemove, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdVirtualBackgroundReplace, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdAutoFrame,
     "usable with degraded behavior; unavailable on the analyzed machine",
     "vulkan_auto_frame_yunet_unavailable"},
    {kEffectIdEyeContact, "diagnostics-only / stub-unavailable",
     "open_vulkan_eye_contact_unavailable"},
    {kEffectIdVideoNoiseRemoval, "diagnostics-only / stub-unavailable",
     "open_vulkan_video_noise_removal_unavailable"},
    {kEffectIdVirtualKeyLight, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdVignette, "production usable (fixed center only)",
     "open_vulkan_vignette_fixed_center_production_ready"},
}};

std::string ReadFile(const fs::path &path) {
  std::ifstream in(path);
  if (!in)
    return {};
  std::ostringstream contents;
  contents << in.rdbuf();
  return contents.str();
}

bool Require(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

bool Contains(const std::string &text, std::string_view needle) {
  return text.find(needle) != std::string::npos;
}

std::string Section(const std::string &text, std::string_view begin,
                    std::string_view end) {
  const std::size_t first = text.find(begin);
  if (first == std::string::npos)
    return {};
  const std::size_t last = text.find(end, first + begin.size());
  return text.substr(first, last == std::string::npos ? std::string::npos
                                                      : last - first);
}

bool TestAuditCoversCanonicalEffects() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string audit =
      ReadFile(root / "docs" / "INSTALLER_VULKAN_CAPABILITIES.md");
  bool ok =
      Require(!audit.empty(), "Vulkan installer capability audit missing");
  std::set<std::string_view> ids;
  for (const auto &row : kExpectedCapabilities) {
    ok &= Require(ids.insert(row.effect_id).second,
                  "duplicate canonical effect in capability matrix");
    const std::string table_prefix = "| `" + std::string(row.effect_id) +
                                     "` | " + std::string(row.status) + " |";
    ok &= Require(Contains(audit, table_prefix),
                  "audit row/status missing for " + std::string(row.effect_id));
    ok &=
        Require(Contains(audit, "`" + std::string(row.reason_code) + "`"),
                "audit reason code missing for " + std::string(row.effect_id));
  }
  ok &= Require(ids.size() == kExpectedCapabilities.size(),
                "capability matrix must cover exactly nine canonical effects");
  ok &= Require(
      Contains(audit,
               "Mirror and fixed-center vignette are the only canonical "
               "effects currently") &&
          Contains(audit, "`effect.<id>.cpu.no_selectable_production_path`") &&
          Contains(audit, "`available_effects` is useful live-stage evidence "
                          "but is never sufficient"),
      "audit must state the exact per-effect recommendation and CPU gates");
  return ok;
}

bool TestInstallerRecommendationConsumesExactPerEffectEvidence() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string backend = ReadFile(
      root / "installer" / "backend" / "studiocast-installer-backend");
  const std::string diagnostics =
      ReadFile(root / "src" / "core" / "vulkan" / "kernels" /
               "resize_bilinear.cpp");
  const std::string device_header =
      ReadFile(root / "src" / "core" / "vulkan" / "vulkan_device.h");
  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");
  bool ok = Require(!backend.empty() && !diagnostics.empty() &&
                        !device_header.empty() && !daemon.empty(),
                    "installer/Vulkan capability sources are missing");
  ok &= Require(
      Contains(backend, "def common_vulkan_effect_evidence") &&
          Contains(backend, "runtime_diagnostics_available") &&
          Contains(backend, "non_cpu_device_selected") &&
          Contains(backend, "compute_queue_available") &&
          Contains(backend, "context_healthy") &&
          Contains(backend, "utility_kernels_ready") &&
          Contains(backend, "def engine_capability_usable") &&
          Contains(backend,
                   "all(evidence.get(key) is True for key in common_keys)") &&
          Contains(backend, "expected_success in successes.get") &&
          !Contains(backend,
                    "v0.2.9 Vulkan has no production-recommendable canonical "
                    "effect"),
      "recommendation must require common and exact per-effect Vulkan proof");
  ok &= Require(
      Contains(backend, "mirror_production_ready") &&
          Contains(backend, "mirror_readiness_code") &&
          Contains(backend, "vignette_fixed_center_production_ready") &&
          Contains(backend, "vignette_parameter_contract") &&
          Contains(backend, "auto_frame_production_ready") &&
          Contains(backend, "auto_frame_cpu_tail") &&
          Contains(backend, "success_reason_codes") &&
          Contains(backend, "blocker_reason_codes") &&
          Contains(backend, "selectable_production_path") &&
          Contains(backend, "effect.{effect}.cpu.no_selectable_production_path"),
      "per-effect success/blocker/evidence/CPU facts are incomplete");
  ok &= Require(
      Contains(diagnostics, "ApplyOpenVulkanPixelEffectReadiness") &&
          Contains(diagnostics, "ApplyOpenVulkanAutoFrameReleaseReadiness") &&
          Contains(diagnostics, "kOpenVulkanMirrorProductionReadyReason") &&
          Contains(diagnostics,
                   "kOpenVulkanVignetteFixedCenterProductionReadyReason") &&
          Contains(device_header, "mirror_production_ready") &&
          Contains(device_header,
                   "vignette_fixed_center_production_ready") &&
          Contains(device_header, "auto_frame_crop_stage_implemented") &&
          Contains(daemon, "vulkan_backend_disabled_in_build"),
      "daemon ON/OFF per-effect readiness schema is incomplete");
  return ok;
}

bool TestExplicitVulkanFilteringMatchesAudit() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string filtering = Section(
      pipeline, "const auto denoise_compatibility", "const bool vb_requested");
  bool ok = Require(!filtering.empty(),
                    "explicit Vulkan filtering section not found");
  ok &= Require(Contains(filtering,
                         "ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility") &&
                    Contains(filtering, "kEffectIdVideoNoiseRemoval") &&
                    Contains(filtering, "denoise_compatibility.blocker_code"),
                "explicit Vulkan must remove video denoise through its exact "
                "fail-closed contract");
  ok &= Require(
      Contains(filtering,
               "ApplyOpenVulkanEyeContactPlanCompatibility(&plan)") &&
          Contains(filtering, "kEffectIdEyeContact") &&
          Contains(filtering,
                   "eye_contact_compatibility.blocker_code"),
      "explicit Vulkan must remove eye contact through its exact fail-closed "
      "contract");
  ok &= Require(!Contains(filtering, "kEffectIdVignette"),
                "explicit Vulkan must retain production vignette in the "
                "canonical plan");
  ok &= Require(!Contains(filtering, "kEffectIdMirror"),
                "explicit Vulkan must retain mirror in the canonical plan");
  ok &=
      Require(Contains(filtering, "virtual background, virtual key light, and"),
              "explicit Vulkan eligible-effect note changed; audit required");

  const std::string rules = ReadFile(root / "src" / "core" / "video" /
                                     "effects" / "broadcast_effect_rules.cpp");
  ok &= Require(Contains(rules, "kEffectIdMirror") &&
                    Contains(rules, "Mirror is the final visual transform"),
                "mirror must be scheduled as the canonical final transform");
  return ok;
}

bool TestEyeContactDiagnosticsOnlyContractIsFailClosed() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string header = ReadFile(
      root / "src" / "core" / "video" / "open_vulkan_eye_contact.h");
  const std::string implementation = ReadFile(
      root / "src" / "core" / "video" / "open_vulkan_eye_contact.cpp");
  const std::string device_header =
      ReadFile(root / "src" / "core" / "vulkan" / "vulkan_device.h");
  const std::string device =
      ReadFile(root / "src" / "core" / "vulkan" / "vulkan_device.cpp");
  const std::string diagnostics =
      ReadFile(root / "src" / "core" / "vulkan" / "kernels" /
               "resize_bilinear.cpp");
  const std::string pipeline_header =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.h");
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");

  bool ok = Require(!header.empty() && !implementation.empty(),
                    "Vulkan eye-contact diagnostics contract is missing");
  ok &= Require(
      Contains(header, "open_vulkan_eye_contact_unavailable") &&
          Contains(header,
                   "open_vulkan_eye_contact_runtime_unavailable") &&
          Contains(header, "live_stage_implemented = false") &&
          Contains(header, "production_adapter_available = false") &&
          Contains(header,
                   "vulkan_inference_provider_available = false") &&
          Contains(header, "shared_device_imported = false") &&
          Contains(header, "queue_ownership_explicit = false") &&
          Contains(header, "artifact_contract_validated = false") &&
          Contains(header, "selectable_cpu_fallback = false") &&
          Contains(header, "dispatch_count = 0") &&
          Contains(header, "cpu_readback_count = 0") &&
          Contains(header, "cpu_fallback_count = 0") &&
          !Contains(header, " Apply(") && !Contains(header, "VulkanImage"),
      "eye contact must expose separate fail-closed facts without a callable "
      "effect stub");
  ok &= Require(
      Contains(implementation, "CurrentOpenVulkanEyeContactFacts") &&
          Contains(implementation,
                   "EvaluateOpenVulkanEyeContactReadiness") &&
          Contains(implementation, "facts.live_stage_implemented") &&
          Contains(implementation, "facts.dispatch_count > 0") &&
          Contains(implementation, "facts.cpu_readback_count == 0") &&
          Contains(implementation, "facts.cpu_fallback_count == 0"),
      "compiled Vulkan presence must not bypass live/runtime/zero-tail gates");
  ok &= Require(
      Contains(diagnostics, "ApplyOpenVulkanEyeContactReadiness") &&
          Contains(diagnostics,
                   "current_facts.non_cpu_device_selected = "
                   "d->non_cpu_device_selected") &&
          Contains(diagnostics,
                   "current_facts.compute_queue_available = "
                   "d->compute_queue_available") &&
          Contains(diagnostics,
                   "current_facts.context_healthy = d->context_healthy") &&
          Contains(diagnostics, "d->blocked_effects[effect_id]") &&
          Contains(diagnostics, "d->available_effects.erase") &&
          Contains(device_header, "eye_contact_production_ready = false") &&
          Contains(device_header,
                   "eye_contact_live_stage_implemented = false") &&
          Contains(device_header,
                   "eye_contact_non_cpu_device_selected = false") &&
          Contains(device_header,
                   "eye_contact_compute_queue_available = false") &&
          Contains(device_header,
                   "eye_contact_context_healthy = false") &&
          Contains(device_header,
                   "eye_contact_selectable_cpu_fallback = false") &&
          Contains(device, "eye_contact_dispatch_count") &&
          Contains(device, "eye_contact_cpu_readback_count") &&
          Contains(device, "eye_contact_cpu_fallback_count"),
      "default diagnostics must block eye contact and expose zero-work facts");
  ok &= Require(
      Contains(pipeline_header,
               "ApplyOpenVulkanEyeContactPlanCompatibility") &&
          Contains(pipeline,
                   "ApplyOpenVulkanEyeContactPlanCompatibility(&plan)") &&
          Contains(pipeline,
                   "eye_contact_compatibility.reason_code") &&
          Contains(pipeline,
                   "eye_contact_compatibility.blocker_code"),
      "canonical explicit-Vulkan plan must publish the nested eye-contact "
      "blocker");
  ok &= Require(
      Contains(daemon, "open_vulkan_eye_contact_unavailable") &&
          Contains(daemon, "eye_contact_blocker_code") &&
          Contains(daemon, "diag->eye_contact_blocker_code") &&
          Contains(daemon, "diag->eye_contact_detail") &&
          Contains(daemon,
                   "\\\"eye_contact_non_cpu_device_selected\\\":false") &&
          Contains(daemon,
                   "\\\"eye_contact_compute_queue_available\\\":false") &&
          Contains(daemon,
                   "\\\"eye_contact_context_healthy\\\":false") &&
          Contains(daemon, "eye_contact_selectable_cpu_fallback") &&
          Contains(daemon, "eye_contact_dispatch_count"),
      "Vulkan-off status must preserve the same exact per-effect contract");
  return ok;
}

bool TestVideoNoiseRemovalDiagnosticsOnlyContractIsFailClosed() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string header = ReadFile(root / "src" / "core" / "video" /
                                      "open_vulkan_video_noise_removal.h");
  const std::string implementation = ReadFile(
      root / "src" / "core" / "video" / "open_vulkan_video_noise_removal.cpp");
  const std::string denoiser = ReadFile(root / "src" / "core" / "open_video" /
                                        "fastdvdnet_denoiser.cpp");
  const std::string denoiser_header =
      ReadFile(root / "src" / "core" / "open_video" / "fastdvdnet_denoiser.h");
  const std::string ort =
      ReadFile(root / "src" / "core" / "onnx" / "ort_session.cpp");
  const std::string device_header =
      ReadFile(root / "src" / "core" / "vulkan" / "vulkan_device.h");
  const std::string device =
      ReadFile(root / "src" / "core" / "vulkan" / "vulkan_device.cpp");
  const std::string diagnostics = ReadFile(root / "src" / "core" / "vulkan" /
                                           "kernels" / "resize_bilinear.cpp");
  const std::string pipeline_header =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.h");
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");

  bool ok = Require(!header.empty() && !implementation.empty(),
                    "Vulkan video-denoise diagnostics contract is missing");
  ok &= Require(
      Contains(header, "open_vulkan_video_noise_removal_unavailable") &&
          Contains(header,
                   "open_vulkan_video_noise_removal_runtime_unavailable") &&
          Contains(header, "live_stage_implemented = false") &&
          Contains(header, "production_adapter_available = false") &&
          Contains(header, "vulkan_inference_provider_available = false") &&
          Contains(header, "shared_device_imported = false") &&
          Contains(header, "queue_ownership_explicit = false") &&
          Contains(header, "artifact_contract_validated = false") &&
          Contains(header, "fully_device_resident_tensor_io = false") &&
          Contains(header, "device_resident_preprocess = false") &&
          Contains(header, "device_resident_postprocess = false") &&
          Contains(header, "temporal_history_device_resident = false") &&
          Contains(header, "temporal_history_bounded = false") &&
          Contains(header, "history_reset_on_disable = false") &&
          Contains(header, "history_reset_on_reconfigure = false") &&
          Contains(header, "capture_sequence_discontinuity_reset = false") &&
          Contains(header, "selectable_cpu_fallback = false") &&
          Contains(header, "dispatch_count = 0") &&
          Contains(header, "temporal_history_reset_count = 0") &&
          Contains(header, "cpu_readback_count = 0") &&
          Contains(header, "cpu_fallback_count = 0") &&
          !Contains(header, " Apply(") && !Contains(header, "VulkanImage"),
      "video denoise must expose separate fail-closed temporal/runtime facts "
      "without a callable effect stub");
  ok &= Require(
      Contains(implementation, "CurrentOpenVulkanVideoNoiseRemovalFacts") &&
          Contains(implementation,
                   "EvaluateOpenVulkanVideoNoiseRemovalReadiness") &&
          Contains(implementation, "facts.live_stage_implemented") &&
          Contains(implementation, "facts.temporal_history_bounded") &&
          Contains(implementation, "facts.history_reset_on_disable") &&
          Contains(implementation, "facts.history_reset_on_reconfigure") &&
          Contains(implementation,
                   "facts.capture_sequence_discontinuity_reset") &&
          Contains(implementation, "facts.dispatch_count > 0") &&
          Contains(implementation, "facts.cpu_readback_count == 0") &&
          Contains(implementation, "facts.cpu_fallback_count == 0"),
      "compiled Vulkan presence must not bypass live/runtime/temporal gates");
  ok &= Require(
      Contains(denoiser, "kFastDvdnetWindowFrames = 5") &&
          Contains(denoiser, "kFastDvdnetHistoryFrames = 3") &&
          Contains(denoiser, "f_tp1 = f_t0") &&
          Contains(denoiser, "f_tp2 = f_t0") &&
          Contains(denoiser_header,
                   "std::vector<std::vector<float>> history_") &&
          Contains(denoiser,
                   "capture_sequence != last_capture_sequence_ + 1") &&
          Contains(denoiser, "if (s <= 0)") &&
          Contains(denoiser, "history_.clear()") &&
          Contains(denoiser, "active_requested_model_id_") &&
          Contains(denoiser, "ResetTemporalState()") &&
          Contains(denoiser, "PreprocessRgbToChwPadded") &&
          Contains(denoiser, "PostprocessToRgbInPlace") &&
          Contains(denoiser, "UploadFromCpuF32") &&
          Contains(denoiser, "DownloadToCpuF32") &&
          Contains(denoiser, "switched to CPU fallback") &&
          !Contains(ort, "VulkanExecutionProvider"),
      "existing FastDVDnet/ORT evidence must remain explicitly host/CUDA/CPU, "
      "not Vulkan-native");
  ok &= Require(
      Contains(diagnostics, "ApplyOpenVulkanVideoNoiseRemovalReadiness") &&
          Contains(diagnostics, "current_facts.non_cpu_device_selected = "
                                "d->non_cpu_device_selected") &&
          Contains(diagnostics, "current_facts.compute_queue_available = "
                                "d->compute_queue_available") &&
          Contains(diagnostics,
                   "current_facts.context_healthy = d->context_healthy") &&
          Contains(diagnostics, "d->blocked_effects[effect_id]") &&
          Contains(diagnostics, "d->available_effects.erase") &&
          Contains(device_header,
                   "video_noise_removal_production_ready = false") &&
          Contains(device_header,
                   "video_noise_removal_non_cpu_device_selected = false") &&
          Contains(device_header,
                   "video_noise_removal_compute_queue_available = false") &&
          Contains(device_header,
                   "video_noise_removal_context_healthy = false") &&
          Contains(device_header,
                   "video_noise_removal_temporal_history_bounded = false") &&
          Contains(device_header,
                   "video_noise_removal_selectable_cpu_fallback = false") &&
          Contains(device, "video_noise_removal_dispatch_count") &&
          Contains(device,
                   "video_noise_removal_temporal_history_reset_count") &&
          Contains(device, "video_noise_removal_cpu_readback_count") &&
          Contains(device, "video_noise_removal_cpu_fallback_count"),
      "default diagnostics must block video denoise and expose zero-work "
      "temporal facts");
  ok &= Require(
      Contains(pipeline_header,
               "ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility") &&
          Contains(pipeline,
                   "ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(") &&
          Contains(pipeline, "denoise_compatibility.reason_code") &&
          Contains(pipeline, "denoise_compatibility.blocker_code"),
      "canonical explicit-Vulkan plan must publish the nested video-denoise "
      "blocker");
  ok &= Require(
      Contains(daemon, "open_vulkan_video_noise_removal_unavailable") &&
          Contains(daemon, "video_noise_removal_blocker_code") &&
          Contains(daemon, "diag->video_noise_removal_blocker_code") &&
          Contains(daemon, "diag->video_noise_removal_detail") &&
          Contains(daemon,
                   "\\\"video_noise_removal_non_cpu_device_selected\\\":"
                   "false") &&
          Contains(daemon,
                   "\\\"video_noise_removal_compute_queue_available\\\":"
                   "false") &&
          Contains(daemon,
                   "\\\"video_noise_removal_context_healthy\\\":false") &&
          Contains(daemon, "video_noise_removal_selectable_cpu_fallback") &&
          Contains(daemon, "video_noise_removal_temporal_history_reset_count"),
      "Vulkan-off status must preserve the exact per-effect temporal contract");
  return ok;
}

bool TestVignetteProductionHelperIsCalledAtTheLiveFinalBoundary() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string orchestration = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_final_resident_stage.cpp");
  const std::string executable_test = ReadFile(
      root / "tests" / "vulkan_final_resident_stage_integration_tests.cpp");
  const std::string cmake = ReadFile(root / "CMakeLists.txt");
  const std::size_t input = pipeline.find(
      "OpenVulkanFinalResidentStageInput final_input");
  const std::size_t vignette_request = pipeline.find(
      "final_input.request_fixed_center_vignette", input);
  const std::size_t ordering_boundary = pipeline.find(
      "final_input.preceding_effects_complete = true", vignette_request);
  const std::size_t apply = pipeline.find(
      "ExecuteOpenVulkanResidentFrameFinalStage(", ordering_boundary);
  const std::size_t download =
      pipeline.find("DownloadImageToRgb(*download_img", apply);
  const std::size_t auto_frame_setup = pipeline.find("// Auto Frame.");
  const std::size_t auto_frame_resolution = pipeline.find(
      "If Auto Frame couldn't be initialized on any backend", auto_frame_setup);
  const std::size_t compatibility_call = pipeline.find(
      "ApplyOpenVulkanVignettePlanCompatibility(fx, &plan)", auto_frame_setup);
  const std::size_t vignette_setup =
      pipeline.find("// Vignette final-output stage.", compatibility_call);
  const std::size_t compatibility_status_publish =
      pipeline.find("append_rule_notes();", compatibility_call);
  bool ok = Require(
      input != std::string::npos && vignette_request != std::string::npos &&
          ordering_boundary != std::string::npos &&
          apply != std::string::npos && download != std::string::npos &&
          auto_frame_setup != std::string::npos &&
          auto_frame_resolution != std::string::npos &&
          compatibility_call != std::string::npos &&
          vignette_setup != std::string::npos &&
          compatibility_status_publish != std::string::npos,
      "vignette live final-output helper call is incomplete");
  ok &= Require(
      input < vignette_request && vignette_request < ordering_boundary &&
          ordering_boundary < apply &&
          apply < download && auto_frame_setup < auto_frame_resolution &&
          auto_frame_resolution < compatibility_call &&
          compatibility_call < vignette_setup &&
          vignette_setup < compatibility_status_publish,
      "live path must batch output resize, vignette, optional mirror, "
      "then perform the final readback, with compatibility evaluated "
      "after Auto Frame setup and its stable block published to status");
  ok &= Require(
      Contains(orchestration, "OpenVulkanVignetteFinalStageInput stage") &&
          Contains(orchestration, "stage.resize_scratch") &&
          Contains(orchestration, "stage.mirrored_dst") &&
          Contains(orchestration, "resources.vignette->ApplyFinal") &&
          Contains(executable_test, "RunProductionCases") &&
          Contains(executable_test,
                   "RunAutoFrameFinalOrchestrationCases") &&
          Contains(executable_test, "RunOrderingAndIsolationCases") &&
          Contains(executable_test, "STUDIOCAST_REQUIRE_VULKAN_RUNTIME") &&
          Contains(cmake,
                   "studiocast-vulkan-final-stage-integration-tests"),
      "vignette production evidence requires the executable test of the exact "
      "CameraPipeline final resident helper");
  ok &= Require(
      Contains(pipeline, "have_open_vulkan_vignette") &&
          Contains(pipeline,
                   "Open Vulkan: Vignette (fixed center after framing") &&
          Contains(pipeline, "DEVICE_LOCAL source/scratch/output and") &&
          Contains(pipeline, "attenuation mask, bounded upload/final-readback") &&
          !Contains(pipeline, "device-resident radial mask") &&
          Contains(pipeline,
                   "open_vulkan_vignette_counters.factor_generation_calls") &&
          Contains(pipeline,
                   "open_vulkan_vignette_counters.factor_upload_calls") &&
          Contains(pipeline, "set_backend(stage_id, \"open_vulkan\")"),
      "vignette backend, setup accounting, or fixed-center evidence is "
      "missing");

  const std::string compatibility =
      Section(pipeline,
              "ApplyOpenVulkanVignettePlanCompatibility(\n"
              "    const effects::BroadcastCameraEffects &fx",
              "namespace {");
  ok &= Require(
      !compatibility.empty() && Contains(compatibility, "kEffectIdAutoFrame") &&
          Contains(compatibility, "kEffectIdVignette") &&
          Contains(compatibility, "ordered_effect_ids.erase") &&
          Contains(compatibility, "disabled.push_back") &&
          Contains(compatibility,
                   "kOpenVulkanVignetteTrackedCenterNotSupportedReason") &&
          !Contains(compatibility, "kEffectIdMirror"),
      "tracked-center compatibility must remove and visibly block only "
      "vignette after confirming retained Auto Frame");
  const std::string pipeline_header =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.h");
  ok &=
      Require(Contains(pipeline_header,
                       "vulkan_vignette_tracked_center_not_supported") &&
                  Contains(pipeline_header,
                           "tracked-center semantics require a device-resident "
                           "center/mask ") &&
                  Contains(pipeline_header, "\"implementation\""),
              "tracked-center compatibility stable reason/detail is missing");

  const std::string helper =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_vignette.cpp");
  const std::string helper_header =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_vignette.h");
  ok &= Require(Contains(helper, "OpenVulkanVignetteRuntimeFailure") &&
                    Contains(helper, "ApplyFinalVignetteU8x3") &&
                    Contains(helper, "production_hardware_ready") &&
                    Contains(helper, "UploadF32_1ToDeviceLocal") &&
                    Contains(helper, "/*map_memory=*/false") &&
                    Contains(helper_header,
                             "vulkan_effect_residency_contract_failed") &&
                    Contains(helper, "factor_upload_calls") &&
                    Contains(helper, "0.70710677f"),
                "vignette stable reasons, hardware gate, CUDA coordinate "
                "contract, or real kernel call is missing");
  const std::string utility = ReadFile(root / "src" / "core" / "vulkan" /
                                       "kernels" / "utility_kernels.cpp");
  ok &= Require(Contains(utility, "output_resize") &&
                    Contains(utility, "final_vignette") &&
                    Contains(utility, "final_mirror") &&
                    Contains(utility, "Op::composite_solid") &&
                    Contains(utility, "no small-alpha early return") &&
                    Contains(utility, "VulkanBufferAccess::compute_write") &&
                    Contains(utility, "frame_batch_.Complete"),
                "vignette final batch must retain dependency barriers and a "
                "single completion seam");
  return ok;
}

bool TestMirrorProductionHelperIsCalledAtTheLiveFinalBoundary() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string orchestration = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_final_resident_stage.cpp");
  const std::string executable_test = ReadFile(
      root / "tests" / "vulkan_final_resident_stage_integration_tests.cpp");
  const std::string cmake = ReadFile(root / "CMakeLists.txt");
  const std::size_t combined_input = pipeline.find(
      "OpenVulkanFinalResidentStageInput final_input");
  const std::size_t analysis_boundary = pipeline.find(
      "final_input.unmirrored_analysis_complete = true", combined_input);
  const std::size_t apply = pipeline.find(
      "ExecuteOpenVulkanResidentFrameFinalStage(", analysis_boundary);
  const std::size_t download =
      pipeline.find("DownloadImageToRgb(*download_img", apply);
  bool ok =
      Require(combined_input != std::string::npos &&
                  analysis_boundary != std::string::npos &&
                  apply != std::string::npos && download != std::string::npos,
              "mirror live final-output helper call is incomplete");
  ok &= Require(combined_input < analysis_boundary &&
                    analysis_boundary < apply && apply < download,
                "live path must call the canonical final resident helper and "
                "mirror before readback");
  ok &= Require(
      Contains(pipeline, "have_open_vulkan_mirror") &&
          Contains(pipeline, "final_input.request_mirror") &&
          Contains(pipeline, "final_result.mirror_applied") &&
          Contains(pipeline, "set_backend(stage_id, \"open_vulkan\")") &&
          Contains(pipeline, "open_vulkan_mirror_counters.dispatch_calls"),
      "mirror batching, applied-backend, or counter evidence is missing");
  ok &= Require(
      Contains(orchestration, "OpenVulkanMirrorResizeFinalStageInput stage") &&
          Contains(orchestration, "OpenVulkanMirrorFinalStageInput stage") &&
          Contains(orchestration, "resources.mirror->ApplyResizeFinal") &&
          Contains(orchestration, "resources.mirror->ApplyFinal") &&
          Contains(executable_test, "mirror RGB no resize") &&
          Contains(executable_test, "mirror BGR resize") &&
          Contains(executable_test, "input_image.device_local()") &&
          Contains(executable_test,
                   "host_visible_intermediate_allocation_calls") &&
          Contains(executable_test, "ReadbackU8x3") &&
          Contains(executable_test, "STUDIOCAST_REQUIRE_VULKAN_RUNTIME") &&
          Contains(cmake,
                   "studiocast-vulkan-final-stage-integration-tests"),
      "mirror production evidence requires the executable test of the exact "
      "CameraPipeline final resident helper");

  const std::string helper =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_mirror.cpp");
  const std::string helper_header =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_mirror.h");
  ok &= Require(Contains(helper, "vulkan_effect_initialization_failed") ||
                    Contains(ReadFile(root / "src" / "core" / "video" /
                                      "open_vulkan_mirror.h"),
                             "vulkan_effect_initialization_failed"),
                "mirror initialization stable reason is missing");
  ok &= Require(Contains(helper, "OpenVulkanMirrorRuntimeFailure") &&
                    Contains(helper, "MirrorHorizontalU8x3") &&
                    Contains(helper, "ResizeMirrorHorizontalU8x3") &&
                    Contains(helper_header,
                             "vulkan_effect_residency_contract_failed") &&
                    Contains(helper, "non-mapped DEVICE_LOCAL resources") &&
                    Contains(helper, "production_hardware_ready"),
                "mirror runtime stable reason or real kernel call is missing");
  const std::string utility = ReadFile(root / "src" / "core" / "vulkan" /
                                       "kernels" / "utility_kernels.cpp");
  ok &= Require(Contains(utility, "output_resize") &&
                    Contains(utility, "final_mirror") &&
                    Contains(utility, "VulkanBufferAccess::compute_write") &&
                    Contains(utility, "frame_batch_.Complete"),
                "mirror resize batch must retain its dependency barrier and "
                "single completion seam");
  return ok;
}

bool TestVirtualBackgroundBlurUsesTheCanonicalResidentWrapper() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string setup = Section(pipeline, "// Background effects.",
                                    "// Auto Frame.");
  const std::string live = Section(
      pipeline,
      "if (stage_id == studiocast::video::effects::contract::\n"
      "                            kEffectIdVirtualBackgroundBlur ||",
      "if (have_maxine_bg_blur)");
  const std::string vulkan_context = Section(
      pipeline, "struct OpenVulkanVirtualBackgroundContext",
      "} open_vulkan_vb;");
  const std::string resident_blur = Section(
      vulkan_context,
      "if (fx.virtual_background.mode == VirtualBackgroundMode::blur)",
      "else if (fx.virtual_background.mode == VirtualBackgroundMode::remove)");
  const std::string matting_setup =
      Section(vulkan_context,
              "bool EnsureInitialized(\n        int frame_w, int frame_h",
              "bool EnsureMatteForFrameGpu(");
  bool ok = Require(!setup.empty() && !live.empty() &&
                        !vulkan_context.empty() && !resident_blur.empty() &&
                        !matting_setup.empty(),
                    "Vulkan blur setup/live resident sections are missing");
  ok &= Require(
      Contains(setup, "open_vulkan_vb.EnsureInitialized") &&
          Contains(setup,
                   "OpenVulkanVirtualBackgroundBlurInitializationFailure") &&
          Contains(setup, "remove_stage_from_plan(*vb_effect_id)"),
      "blur setup must retain the stage only after shared matting/wrapper "
      "readiness and publish a stable outer failure");
  ok &= Require(
      Contains(resident_blur, "blur_effect.Apply") &&
          Contains(resident_blur, "matting_session->Readiness()") &&
          Contains(resident_blur, "cached_frame_alpha_sequence") &&
          Contains(resident_blur, "alpha_resize_completion_count") &&
          !Contains(resident_blur, "BoxBlurSeparableF32_1") &&
          !Contains(resident_blur, "BoxBlurCompositeAlphaU8x3") &&
          !Contains(resident_blur, "Readback") &&
          !Contains(resident_blur, "Cpu"),
      "canonical live blur mode must route the complete resident chain only "
      "through its effect wrapper with current matte evidence");
  ok &= Require(
      Contains(live, "OptionalEffectSlot::virtual_background") &&
          Contains(live, "vulkan_vb_breaker.AllowsAttempt") &&
          Contains(live, "block_optional_effect(") &&
          Contains(live, "clear_optional_effect_on_success") &&
          Contains(live, "OpenVulkanVirtualBackgroundBlurRuntimeFailure"),
      "Vulkan blur runtime failures must use the existing bounded breaker and "
      "stable effect reason instead of retrying every frame");

  const std::string readback_allocation = Section(
      matting_setup, "if (require_cpu_alpha_readback)",
      "if (require_vb_buffers)");
  ok &= Require(
      !readback_allocation.empty() &&
          Contains(readback_allocation, "alpha_readback") &&
          Contains(readback_allocation, "alpha_cpu.resize") &&
          Contains(readback_allocation,
                   "vulkan_auto_frame_alpha_readback_setup_failed") &&
          !Contains(readback_allocation, "LatchExternalFailure") &&
          !Contains(readback_allocation, "matting_failure_latched = true") &&
          Contains(pipeline, "/*require_cpu_alpha_readback=*/true") &&
          Contains(pipeline,
                   "Open Vulkan Auto Frame: matte alpha readback failed") &&
          !Contains(resident_blur, "require_cpu_alpha_readback"),
      "host alpha staging must be explicit Auto Frame-only setup and must not "
      "poison shared matting or appear in blur");
  const std::string resident_allocations = Section(
      matting_setup, "if (!EnsureImage(&frame_rgb",
      "if (require_cpu_alpha_readback)");
  ok &= Require(!resident_allocations.empty() &&
                    !Contains(resident_allocations, "LatchExternalFailure") &&
                    !Contains(resident_allocations,
                              "matting_failure_latched = true"),
                "consumer/transport/scratch allocation failures must not "
                "poison shared matting consumers");

  const std::string wrapper = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_virtual_background_blur.cpp");
  const std::string wrapper_header = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_virtual_background_blur.h");
  ok &= Require(
      Contains(wrapper, "VulkanMattingReadiness") &&
          Contains(wrapper, "runtime.inference_count == 0") &&
          Contains(wrapper, "runtime.completion_count") &&
          Contains(wrapper, "runtime.cpu_readback_count != 0") &&
          Contains(wrapper, "VulkanPixelFormat::rgb_u8") &&
          Contains(wrapper, "non-mapped DEVICE_LOCAL") &&
          Contains(wrapper, "left->buffer() == right->buffer()") &&
          Contains(wrapper, "BoxBlurSeparableF32_1") &&
          Contains(wrapper, "BoxBlurCompositeAlphaU8x3") &&
          !Contains(wrapper, ".Invalidate(") &&
          !Contains(wrapper, "ReadbackF32_1") &&
          Contains(wrapper_header,
                   "vulkan_virtual_background_blur_initialization_failed") &&
          Contains(wrapper_header,
                   "vulkan_virtual_background_blur_runtime_failed") &&
          Contains(wrapper_header, "alpha_readback_calls = 0") &&
          Contains(wrapper_header, "cpu_fallback_calls = 0"),
      "blur wrapper must enforce current matting, exact RGB/device-local "
      "scratch, alias, stable reason, and zero CPU/readback contracts");

  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");
  ok &= Require(
      Contains(daemon, "virtual_background_blur_dispatch_calls") &&
          Contains(daemon,
                   "virtual_background_blur_alpha_readback_calls") &&
          Contains(daemon, "virtual_background_blur_cpu_fallback_calls") &&
          Contains(daemon,
                   "virtual_background_blur_runtime_failure_frames") &&
          Contains(daemon, "virtual_background_blur_device_loss_frames"),
      "debug diagnostics must expose blur dispatch, zero-host-tail, runtime, "
      "and device-loss counters");
  return ok;
}

bool TestVirtualBackgroundRemoveUsesTheCanonicalResidentWrapper() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string setup = Section(pipeline, "// Background effects.",
                                    "// Auto Frame.");
  const std::string live = Section(
      pipeline,
      "if (stage_id == studiocast::video::effects::contract::\n"
      "                            kEffectIdVirtualBackgroundBlur ||",
      "if (have_maxine_bg_blur)");
  const std::string vulkan_context = Section(
      pipeline, "struct OpenVulkanVirtualBackgroundContext",
      "} open_vulkan_vb;");
  const std::string resident_remove = Section(
      vulkan_context,
      "else if (fx.virtual_background.mode == VirtualBackgroundMode::remove)",
      "else if (fx.virtual_background.mode == VirtualBackgroundMode::replace)");
  const std::string matting_setup =
      Section(vulkan_context,
              "bool EnsureInitialized(\n        int frame_w, int frame_h",
              "bool EnsureMatteForFrameGpu(");
  bool ok = Require(!setup.empty() && !live.empty() &&
                        !vulkan_context.empty() && !resident_remove.empty() &&
                        !matting_setup.empty(),
                    "Vulkan remove setup/live resident sections are missing");
  ok &= Require(
      Contains(setup, "open_vulkan_vb.EnsureInitialized") &&
          Contains(
              setup,
              "OpenVulkanVirtualBackgroundRemoveInitializationFailure") &&
          Contains(setup, "remove_stage_from_plan(*vb_effect_id)"),
      "remove setup must retain the stage only after shared matting/wrapper "
      "readiness and publish a stable outer failure");
  ok &= Require(
      Contains(resident_remove, "remove_effect.Apply") &&
          Contains(resident_remove, "matting_session->Readiness()") &&
          Contains(resident_remove, "cached_frame_alpha_sequence") &&
          Contains(resident_remove, "alpha_resize_completion_count") &&
          !Contains(resident_remove, "BoxBlurSeparableF32_1") &&
          !Contains(resident_remove, "CompositeAlphaSolidU8x3") &&
          !Contains(resident_remove, "ParseRgbHex") &&
          !Contains(resident_remove, "Readback") &&
          !Contains(resident_remove, "Cpu"),
      "canonical live remove mode must route the complete resident chain only "
      "through its effect wrapper with current matte evidence");
  ok &= Require(
      Contains(live, "is_vulkan_remove") &&
          Contains(live, "is_wrapped_vulkan_vb") &&
          Contains(live, "vulkan_vb_breaker.AllowsAttempt") &&
          Contains(live, "block_optional_effect(") &&
          Contains(live, "clear_optional_effect_on_success") &&
          Contains(live, "OpenVulkanVirtualBackgroundRemoveRuntimeFailure"),
      "Vulkan remove runtime failures must use the existing bounded breaker "
      "and stable effect reason instead of retrying every frame");

  const std::string readback_allocation = Section(
      matting_setup, "if (require_cpu_alpha_readback)",
      "if (require_vb_buffers)");
  ok &= Require(
      !readback_allocation.empty() &&
          Contains(readback_allocation, "alpha_readback") &&
          Contains(pipeline, "/*require_cpu_alpha_readback=*/true") &&
          !Contains(resident_remove, "require_cpu_alpha_readback"),
      "host alpha staging must remain explicit Auto Frame-only setup and must "
      "not appear in remove");
  const std::string resident_allocations = Section(
      matting_setup, "if (!EnsureImage(&frame_rgb",
      "if (require_cpu_alpha_readback)");
  ok &= Require(!resident_allocations.empty() &&
                    !Contains(resident_allocations, "LatchExternalFailure") &&
                    !Contains(resident_allocations,
                              "matting_failure_latched = true"),
                "remove consumer/scratch allocation failures must not poison "
                "shared matting consumers");

  const std::string wrapper = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_virtual_background_remove.cpp");
  const std::string wrapper_header = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_virtual_background_remove.h");
  ok &= Require(
      Contains(wrapper, "VulkanMattingReadiness") &&
          Contains(wrapper, "runtime.inference_count == 0") &&
          Contains(wrapper, "runtime.completion_count") &&
          Contains(wrapper, "runtime.cpu_readback_count != 0") &&
          Contains(wrapper, "VulkanPixelFormat::rgb_u8") &&
          Contains(wrapper, "non-mapped DEVICE_LOCAL") &&
          Contains(wrapper,
                   "images[left]->buffer() == images[right]->buffer()") &&
          Contains(wrapper, "BoxBlurSeparableF32_1") &&
          Contains(wrapper, "CompositeAlphaSolidU8x3") &&
          !Contains(wrapper, ".Invalidate(") &&
          !Contains(wrapper, "ReadbackF32_1") &&
          Contains(wrapper_header,
                   "vulkan_virtual_background_remove_initialization_failed") &&
          Contains(wrapper_header,
                   "vulkan_virtual_background_remove_runtime_failed") &&
          Contains(wrapper_header, "alpha_readback_calls = 0") &&
          Contains(wrapper_header, "cpu_fallback_calls = 0"),
      "remove wrapper must enforce current matting, exact RGB/device-local "
      "scratch, alias, stable reason, and zero CPU/readback contracts");

  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");
  ok &= Require(
      Contains(daemon, "virtual_background_remove_dispatch_calls") &&
          Contains(daemon,
                   "virtual_background_remove_alpha_readback_calls") &&
          Contains(daemon, "virtual_background_remove_cpu_fallback_calls") &&
          Contains(daemon,
                   "virtual_background_remove_runtime_failure_frames") &&
          Contains(daemon,
                   "virtual_background_remove_device_loss_frames"),
      "debug diagnostics must expose remove dispatch, zero-host-tail, runtime, "
      "and device-loss counters");
  return ok;
}

bool TestVirtualBackgroundReplaceUsesTheCanonicalResidentWrapper() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string setup = Section(pipeline, "// Background effects.",
                                    "// Auto Frame.");
  const std::string live = Section(
      pipeline,
      "if (stage_id == studiocast::video::effects::contract::\n"
      "                            kEffectIdVirtualBackgroundBlur ||",
      "if (have_maxine_bg_blur)");
  const std::string vulkan_context = Section(
      pipeline, "struct OpenVulkanVirtualBackgroundContext",
      "} open_vulkan_vb;");
  const std::string resident_replace = Section(
      vulkan_context,
      "else if (fx.virtual_background.mode == VirtualBackgroundMode::replace)",
      "return true;\n    }\n\n    bool\n    ApplyVulkanRgb");
  const std::string matting_setup =
      Section(vulkan_context,
              "bool EnsureInitialized(\n        int frame_w, int frame_h",
              "bool EnsureMatteForFrameGpu(");
  bool ok = Require(!setup.empty() && !live.empty() &&
                        !vulkan_context.empty() && !resident_replace.empty() &&
                        !matting_setup.empty(),
                    "Vulkan replace setup/live resident sections are missing");
  ok &= Require(
      Contains(pipeline,
               "open_vulkan_vb.ConfigureReplaceBackgroundSource") &&
          Contains(setup, "open_vulkan_vb.EnsureInitialized") &&
          Contains(
              setup,
              "OpenVulkanVirtualBackgroundReplaceInitializationFailure") &&
          Contains(setup, "remove_stage_from_plan(*vb_effect_id)"),
      "replace setup must prepare/stat its source, initialize the resident "
      "wrapper before publishing the backend, and retain a stable failure");
  ok &= Require(
      Contains(matting_setup, "prepared_bg_src") &&
          Contains(matting_setup, "replace_effect.EnsureInitialized") &&
          Contains(matting_setup, "matting_session->Readiness()") &&
          !Contains(matting_setup, "EnsureReplaceBackgroundGpu"),
      "replace setup must pass the prepared source and verified shared "
      "matting readiness to the wrapper without the former lazy GPU cache");
  ok &= Require(
      Contains(resident_replace, "replace_effect.Apply") &&
          Contains(resident_replace, "matting_session->Readiness()") &&
          Contains(resident_replace, "cached_frame_alpha_sequence") &&
          Contains(resident_replace, "alpha_resize_completion_count") &&
          !Contains(resident_replace, "LoadImageRgb24") &&
          !Contains(resident_replace, "UploadRgbToImage") &&
          !Contains(resident_replace, "ResizeBilinear(") &&
          !Contains(resident_replace, "CompositeAlphaU8x3") &&
          !Contains(resident_replace, "Readback") &&
          !Contains(resident_replace, "Cpu"),
      "canonical live replace mode must route only through the resident "
      "wrapper with current matte evidence and no setup/CPU/readback work");
  ok &= Require(
      Contains(live, "is_vulkan_replace") &&
          Contains(live, "is_wrapped_vulkan_vb") &&
          Contains(live, "vulkan_vb_breaker.AllowsAttempt") &&
          Contains(live, "block_optional_effect(") &&
          Contains(live, "clear_optional_effect_on_success") &&
          Contains(live,
                   "OpenVulkanVirtualBackgroundReplaceRuntimeFailure"),
      "Vulkan replace runtime and transport failures must use the existing "
      "bounded virtual-background breaker and stable effect reason");

  const std::string wrapper = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_virtual_background_replace.cpp");
  const std::string wrapper_header = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_virtual_background_replace.h");
  ok &= Require(
      Contains(wrapper, "runtime.inference_count == 0") &&
          Contains(wrapper, "runtime.completion_count") &&
          Contains(wrapper, "runtime.cpu_readback_count != 0") &&
          Contains(wrapper, "active_device.ownership_domain") &&
          Contains(wrapper, "UploadRgb24ToDeviceLocal") &&
          Contains(wrapper, "LoadImageRgb24") &&
          Contains(wrapper, "if (!source_cache_hit)") &&
          Contains(wrapper, "ResizeBilinear(source_rgb_, replacement_rgb_") &&
          Contains(wrapper, "CompositeAlphaU8x3") &&
          Contains(wrapper, "replacement_rgb_.mapped()") &&
          Contains(wrapper, "replacement_rgb_.device_local()") &&
          !Contains(wrapper, ".Invalidate(") &&
          !Contains(wrapper, "ReadbackF32_1") &&
          Contains(wrapper_header,
                   "vulkan_virtual_background_replace_asset_invalid") &&
          Contains(wrapper_header,
                   "vulkan_virtual_background_replace_asset_upload_failed") &&
          Contains(wrapper_header, "alpha_readback_calls = 0") &&
          Contains(wrapper_header, "cpu_fallback_calls = 0"),
      "replace wrapper must enforce exact current shared-device matting, "
      "setup-only cached asset upload/resize, resident composition, stable "
      "asset reasons, and zero CPU/readback counters");

  const std::string utility =
      ReadFile(root / "src" / "core" / "vulkan" / "kernels" /
               "utility_kernels.cpp");
  const std::string upload = Section(
      utility, "bool UtilityKernels::UploadRgb24ToDeviceLocal(",
      "bool UtilityKernels::ReadbackF32_1(");
  ok &= Require(
      !upload.empty() && Contains(upload, "VulkanPixelFormat::rgb_u8") &&
          Contains(upload,
                   "upload_staging.byte_size() != device_dst.byte_size()") &&
          Contains(upload, "VulkanBufferAccess::host_write") &&
          Contains(upload, "VulkanBufferAccess::transfer_read") &&
          Contains(upload, "vkCmdCopyBuffer") &&
          Contains(upload, "VulkanBufferAccess::transfer_write") &&
          Contains(upload, "VulkanBufferAccess::compute_read") &&
          Contains(upload, "SubmitRecorded(error_out)"),
      "setup upload must require exact RGB/byte identity and explicit "
      "host-transfer-compute synchronization on the shared queue");

  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");
  ok &= Require(
      Contains(daemon,
               "virtual_background_replace_asset_allocation_calls") &&
          Contains(daemon,
                   "virtual_background_replace_asset_decode_calls") &&
          Contains(daemon,
                   "virtual_background_replace_asset_upload_calls") &&
          Contains(daemon,
                   "virtual_background_replace_asset_resize_dispatch_calls") &&
          Contains(daemon, "virtual_background_replace_dispatch_calls") &&
          Contains(daemon,
                   "virtual_background_replace_alpha_readback_calls") &&
          Contains(daemon,
                   "virtual_background_replace_cpu_fallback_calls") &&
          Contains(daemon,
                   "virtual_background_replace_runtime_failure_frames") &&
          Contains(daemon,
                   "virtual_background_replace_device_loss_frames"),
      "debug diagnostics must expose replace setup, dispatch, zero-host-tail, "
      "runtime, and device-loss counters");
  return ok;
}

bool TestVirtualKeyLightUsesTheCanonicalResidentWrapper() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string setup = Section(pipeline, "// Virtual Key Light.",
                                    "// If key light couldn't be initialized");
  const std::string live =
      Section(pipeline, "if (have_open_vulkan_key_light) {",
              "if (have_open_cuda_key_light)");
  const std::string key_light_context = Section(
      pipeline, "struct OpenVulkanKeyLightContext", "} open_vulkan_key_light;");
  const std::string matting_context =
      Section(pipeline, "struct OpenVulkanVirtualBackgroundContext",
              "} open_vulkan_vb;");
  const std::string reuse_check =
      Section(matting_context, "bool HasCompatibleFrameAlpha(",
              "bool EnsureInitialized(");
  bool ok =
      Require(!setup.empty() && !live.empty() && !key_light_context.empty() &&
                  !matting_context.empty() && !reuse_check.empty(),
              "Vulkan key-light setup/live resident sections are "
              "missing");
  ok &= Require(
      Contains(setup, "open_vulkan_key_light.EnsureInitialized") &&
          Contains(setup, "OpenVulkanVirtualKeyLightInitializationFailure") &&
          Contains(setup, "remove_stage_from_plan(stage_id)") &&
          Contains(setup, "set_backend(stage_id, \"open_vulkan\")"),
      "key-light setup must publish the backend only after shared "
      "matting/wrapper readiness and retain a stable outer failure");
  ok &= Require(
      Contains(key_light_context, "key_light_effect.EnsureInitialized") &&
          Contains(key_light_context, "key_light_effect.Apply") &&
          Contains(key_light_context, "HasCompatibleFrameAlpha") &&
          Contains(key_light_context, "inference_count_before") &&
          Contains(key_light_context, "resize_count_before") &&
          Contains(key_light_context, "reused_same_frame") &&
          Contains(key_light_context, "independently_inferred") &&
          !Contains(key_light_context, ".ApplyKeyLightU8x3") &&
          !Contains(key_light_context, "GetAlphaCpuForFrame") &&
          !Contains(key_light_context, "ReadbackF32_1") &&
          !Contains(key_light_context, "require_cpu_alpha_readback"),
      "canonical live key light must route only through its wrapper, classify "
      "exact same-frame reuse versus independent inference, and own no CPU "
      "alpha path");
  ok &= Require(
      Contains(reuse_check,
               "IsOpenVulkanVirtualKeyLightSameFrameArtifactCompatible") &&
          Contains(reuse_check, "&alpha_model") &&
          Contains(reuse_check, "alpha_model.buffer()") &&
          Contains(reuse_check, "alpha_resized.context_identity()") &&
          Contains(reuse_check, "alpha_resized.device_local()"),
      "same-frame reuse must reject stale, foreign, incompatible, or tampered "
      "artifact handles instead of trusting a cache-key hit alone");
  ok &= Require(
      Contains(live, "OptionalEffectSlot::relight") &&
          Contains(live, "key_light_breaker.AllowsAttempt") &&
          Contains(live, "key_light_failures_before") &&
          Contains(live, "key_light_device_losses_before") &&
          Contains(live, "VulkanContextHealth::device_lost") &&
          Contains(live, "block_optional_effect(") &&
          Contains(live, "clear_optional_effect_on_success") &&
          Contains(live, "OpenVulkanVirtualKeyLightRuntimeFailure"),
      "Vulkan key-light setup/transport/runtime failures must be isolated to "
      "the relight breaker, classified by device health, and avoid counter "
      "double-counting");

  const std::string wrapper = ReadFile(root / "src" / "core" / "video" /
                                       "open_vulkan_virtual_key_light.cpp");
  const std::string wrapper_header = ReadFile(
      root / "src" / "core" / "video" / "open_vulkan_virtual_key_light.h");
  ok &= Require(
      Contains(wrapper, "std::clamp(intensity_percent, 0, 100)") &&
          Contains(wrapper, "std::clamp(direction_pan_degrees, -90, 90)") &&
          Contains(wrapper, "target_g = 242.0f") &&
          Contains(wrapper, "target_b = 228.0f") &&
          Contains(wrapper, "target_r = 228.0f") &&
          Contains(wrapper, "runtime.inference_count") &&
          Contains(wrapper, "runtime.completion_count") &&
          Contains(wrapper, "runtime.cpu_readback_count != 0") &&
          Contains(wrapper, "artifact->key == key") &&
          Contains(wrapper, "key.provider_id == \"open_vulkan\"") &&
          Contains(wrapper, "key.model_id == active_model_id") &&
          Contains(wrapper, "FrameMatteStorage::device_f32_alpha") &&
          Contains(wrapper, "key.device_context == device_context") &&
          Contains(wrapper, "key.stream == queue") &&
          Contains(wrapper, "artifact->handle == expected_object_handle") &&
          Contains(wrapper, "artifact->aux_handle == expected_buffer_handle") &&
          Contains(wrapper, "non-mapped DEVICE_LOCAL") &&
          Contains(wrapper, "parameters_.passthrough()") &&
          Contains(wrapper, "ApplyKeyLightU8x3") &&
          !Contains(wrapper, ".Invalidate(") &&
          !Contains(wrapper, "ReadbackF32_1") &&
          Contains(wrapper_header,
                   "vulkan_virtual_key_light_initialization_failed") &&
          Contains(wrapper_header, "vulkan_virtual_key_light_runtime_failed") &&
          Contains(wrapper_header, "alpha_readback_calls = 0") &&
          Contains(wrapper_header, "cpu_fallback_calls = 0"),
      "key-light wrapper must enforce exact parity/current "
      "matting/device-local "
      "contracts, true zero-intensity pass-through, stable reasons, and zero "
      "CPU/readback behavior");

  const std::string ordering =
      ReadFile(root / "src" / "core" / "video" / "effects" /
               "broadcast_effect_rules.cpp");
  const std::size_t vb = ordering.find("if (enable_virtual_background)");
  const std::size_t key = ordering.find("if (enable_key_light)", vb);
  const std::size_t frame = ordering.find("if (enable_auto_frame)", key);
  ok &= Require(vb != std::string::npos && key != std::string::npos &&
                    frame != std::string::npos && vb < key && key < frame,
                "canonical ordering must remain virtual background, key "
                "light, then Auto Frame for same-frame matte reuse");

  const std::string daemon =
      ReadFile(root / "src" / "daemon" / "studiocastd_main.cpp");
  ok &= Require(
      Contains(daemon, "virtual_key_light_shared_matte_reuse_calls") &&
          Contains(daemon,
                   "virtual_key_light_independent_matte_inference_calls") &&
          Contains(daemon, "virtual_key_light_passthrough_frames") &&
          Contains(daemon, "virtual_key_light_alpha_readback_calls") &&
          Contains(daemon, "virtual_key_light_cpu_fallback_calls") &&
          Contains(daemon, "virtual_key_light_runtime_failure_frames") &&
          Contains(daemon, "virtual_key_light_device_loss_frames"),
      "debug diagnostics must expose key-light reuse, independent inference, "
      "pass-through, zero-host-tail, runtime, and device-loss counters");
  return ok;
}

bool TestDefaultMattingDiagnosticsRemainFailClosed() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string diagnostics = ReadFile(root / "src" / "core" / "vulkan" /
                                           "kernels" / "resize_bilinear.cpp");
  const std::string default_diagnostics =
      Section(diagnostics, "OpenVulkanDiagnostics DiagnoseOpenVulkanDefault()",
              "} // namespace studiocast::vulkan");
  bool ok = Require(!default_diagnostics.empty(),
                    "default Open Vulkan diagnostics implementation missing");
  ok &= Require(Contains(default_diagnostics, "d.matting_runtime = \"none\""),
                "default diagnostics no longer report matting runtime none; "
                "audit required");
  ok &= Require(
      Contains(default_diagnostics,
               "d.device_residency_mode = \"unavailable\""),
      "default diagnostics no longer block device-resident matting; audit "
      "required");
  ok &=
      Require(Contains(default_diagnostics, "BlockOpenVulkanVirtualBackground"),
              "default diagnostics no longer block virtual background; "
              "audit required");
  ok &= Require(
      Contains(default_diagnostics, "ApplyOpenVulkanMattingReadiness") &&
          Contains(diagnostics, "EvaluateVulkanMattingReadiness") &&
          Contains(diagnostics, "matting_reason_code"),
      "default diagnostics must consume the shared matting "
      "readiness verdict and stable reason evidence");
  ok &= Require(
      !Contains(diagnostics, "ValidateProductionNcnnVulkanMattingPack") &&
          Contains(diagnostics, "facts.contract_validated = false"),
      "polled diagnostics must not hash artifacts or infer live contract "
      "validation");
  const std::string matting_session =
      ReadFile(root / "src" / "core" / "open_video" /
               "vulkan_matting_session.cpp");
  const std::string camera_pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  ok &= Require(
      Contains(matting_session, "ValidateProductionNcnnVulkanMattingPack") &&
          !Contains(camera_pipeline,
                    "ValidateProductionNcnnVulkanMattingPack"),
      "artifact hashing must remain solely at memoized live-session "
      "validation, not model selection");
  ok &= Require(
      Contains(default_diagnostics, "d.blocked_reason = d.fallback_reason") &&
          Contains(default_diagnostics, "BlockOpenVulkanVirtualBackground(&d, "
                                        "d.blocked_reason.c_str())") &&
          Contains(default_diagnostics,
                   "d.blocked_reason = d.matting_reason_code"),
      "base Vulkan failures must retain blocker precedence while healthy "
      "utility kernels defer to the shared matting blocker");

  const std::string blocker = Section(
      diagnostics, "void BlockOpenVulkanVirtualBackground", "} // namespace");
  ok &= Require(Contains(blocker, "kEffectIdVirtualBackgroundBlur") &&
                    Contains(blocker, "kEffectIdVirtualBackgroundRemove") &&
                    Contains(blocker, "kEffectIdVirtualBackgroundReplace") &&
                    Contains(blocker, "kEffectIdVirtualKeyLight"),
                "matting blocker must cover all virtual-background modes and "
                "virtual key light");
  return ok;
}

bool TestAutoFrameDegradedPathRemainsExplicit() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string auto_frame =
      Section(pipeline, "// Auto Frame.", "// Virtual Key Light.");
  bool ok = Require(!auto_frame.empty(), "Auto Frame setup section missing");
  ok &= Require(
      Contains(auto_frame, "vulkan_auto_frame_cpu_face_tracking_tail") &&
          Contains(auto_frame,
                   "vulkan_auto_frame_cpu_crop_plan_smoothing_tail"),
      "Vulkan Auto Frame CPU face/crop-plan tails changed; audit required");
  ok &= Require(
      Contains(auto_frame,
               "vulkan_auto_frame_matte_alpha_readback_cpu_box_tail"),
      "Vulkan Auto Frame matte readback/CPU-box tail changed; audit required");
  ok &= Require(
      Contains(auto_frame, "YunetProviderPolicy::cpu_only"),
      "explicit Vulkan Auto Frame must force the CPU-only YuNet provider");
  ok &= Require(Contains(auto_frame, "set_backend(stage_id, \"open_vulkan\")"),
                "Vulkan Auto Frame is no longer connected to the live backend");
  const std::size_t setup_call =
      auto_frame.find("open_vulkan_auto_frame.EnsureInitialized");
  const std::size_t have_stage =
      auto_frame.find("have_open_vulkan_auto_frame = true", setup_call);
  const std::size_t set_backend =
      auto_frame.find("set_backend(stage_id, \"open_vulkan\")", have_stage);
  const std::size_t remove_stage =
      auto_frame.find("remove_stage_from_plan(stage_id)", set_backend);
  ok &= Require(
      setup_call != std::string::npos && have_stage != std::string::npos &&
          set_backend != std::string::npos &&
          remove_stage != std::string::npos && setup_call < have_stage &&
          have_stage < set_backend && set_backend < remove_stage,
      "Vulkan Auto Frame must publish the backend only after setup "
      "success and remove the stage on tracking+matting failure");
  ok &= Require(
      Contains(auto_frame, "OpenVulkanAutoFrameFaceProviderFailure(fd_err)") &&
          Contains(auto_frame, "[vulkan_backend_disabled]"),
      "Auto Frame setup failures must expose a stable outer effect reason and "
      "nested provider/backend cause");

  const std::string wrapper =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_auto_frame.cpp");
  ok &= Require(Contains(wrapper, "production_hardware_ready") &&
                    Contains(wrapper, "CropResizeBilinear") &&
                    Contains(wrapper, "vulkan_resource_foreign_context") &&
                    Contains(wrapper, "vulkan_device_lost"),
                "canonical Auto Frame wrapper must enforce production shared "
                "context, crop dispatch, ownership, and device-loss evidence");

  const std::string live_context =
      Section(pipeline, "struct OpenVulkanAutoFrameContext",
              "} open_vulkan_auto_frame;");
  ok &= Require(
      Contains(live_context, "OpenVulkanAutoFrameReuseKeyResetReason") &&
          Contains(live_context,
                   "ExecuteOpenVulkanResidentAutoFrameStage") &&
          !Contains(live_context, ".CropResizeBilinear"),
      "the live Auto Frame context must use the reset/reuse contract and route "
      "crop/resize only through the canonical production wrapper");

  const std::size_t cache_begin =
      pipeline.find("open_video_cache.BeginFrame(capture_sequence)");
  const std::size_t preupload_face =
      pipeline.find("YunetProviderPolicy::cpu_only", cache_begin);
  const std::size_t upload = pipeline.find(
      "ensure_open_vulkan_current_from_cpu(&vk_err)", preupload_face);
  const std::string orchestration = ReadFile(
      root / "src" / "core" / "video" /
      "open_vulkan_final_resident_stage.cpp");
  const std::size_t canonical_crop =
      orchestration.find("auto_frame->ApplyCrop");
  ok &= Require(
      cache_begin != std::string::npos && preupload_face != std::string::npos &&
          upload != std::string::npos && canonical_crop != std::string::npos &&
          cache_begin < preupload_face && preupload_face < upload,
      "Auto Frame must analyze the original host capture before "
      "the one Vulkan upload and use the canonical crop wrapper");
  return ok;
}

bool TestSharedVulkanSynchronizationAndLifetimeContracts() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string utility = ReadFile(root / "src" / "core" / "vulkan" /
                                       "kernels" / "utility_kernels.cpp");
  const std::string device =
      ReadFile(root / "src" / "core" / "vulkan" / "vulkan_device.cpp");
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  bool ok = Require(!utility.empty() && !device.empty() && !pipeline.empty(),
                    "shared Vulkan synchronization sources are missing");

  const std::string generic_dispatch =
      Section(utility, "bool UtilityKernels::DispatchTwoPass(",
              "bool UtilityKernels::SubmitRecorded(");
  ok &= Require(
      Contains(generic_dispatch, "data_read_barriers") &&
          Contains(generic_dispatch,
                   "VulkanBufferAccess::host_or_compute_write") &&
          Contains(generic_dispatch, "VulkanBufferAccess::host_write") &&
          Contains(generic_dispatch, "parameter_read_barrier") &&
          !Contains(generic_dispatch,
                    "std::array<VkBufferMemoryBarrier, 4> read_barriers"),
      "generic utility dispatch must synchronize uploaded or compute-produced "
      "data separately from its exact host-written parameter buffer");

  const std::string safe_destroy =
      Section(device, "bool VulkanContextState::SafeToDestroyChildren() const",
              "VulkanHealthSnapshot VulkanContextState::HealthSnapshot()");
  ok &= Require(
      Contains(safe_destroy, "fatal_wait_completion_unknown") &&
          Contains(safe_destroy,
                   "health.submitted_serial > health.completed_serial") &&
          Contains(safe_destroy, "VulkanContextHealth::unsafe_timeout"),
      "teardown must quarantine timeout and non-timeout fatal wait failures "
      "whose accepted submission has no proven completion");

  const auto verify_submission_order = [&](const std::string &section,
                                           const char *label) {
    const std::size_t reset = section.find("vkResetFences");
    const std::size_t submit = section.find("vkQueueSubmit", reset);
    const std::size_t submitted_serial =
        section.find("++context_->health.submitted_serial", submit);
    const std::size_t wait = section.find("vkWaitForFences", submitted_serial);
    return Require(
        Contains(section, "consume_injected_failure") &&
            Contains(section, "VulkanSubmissionPhase::wait_for_fence") &&
            reset != std::string::npos && submit != std::string::npos &&
            submitted_serial != std::string::npos &&
            wait != std::string::npos && reset < submit &&
            submit < submitted_serial && submitted_serial < wait,
        std::string(label) +
            " must inject at the named phase and record submission only after "
            "vkQueueSubmit succeeds");
  };
  ok &= verify_submission_order(
      Section(device, "bool VulkanDevice::SubmitAndWait(",
              "bool VulkanDevice::CheckDriverResult("),
      "single-command submission");
  ok &= verify_submission_order(Section(device,
                                        "bool VulkanCommandBatch::Complete(",
                                        "void VulkanCommandBatch::Abort()"),
                                "batched submission");

  const std::string vulkan_context =
      Section(pipeline, "struct OpenVulkanVirtualBackgroundContext",
              "} open_vulkan_vb;");
  const std::string matte =
      Section(vulkan_context, "bool EnsureMatteForFrameGpu(",
              "bool EnsureFrameAlphaForFrameGpu(");
  ok &= Require(
      Contains(matte, "utility_submissions_before") &&
          Contains(matte, "utility_submissions_after") &&
          Contains(matte, "runtime_completions_before") &&
          Contains(matte, "runtime_completions_after") &&
          Contains(matte, "forced_sync_calls +=") &&
          !Contains(matte, "++forced_sync_calls"),
      "production Vulkan matting must count observed preprocess and inference "
      "synchronous completions rather than one hardcoded completion");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = TestAuditCoversCanonicalEffects() && ok;
  ok = TestInstallerRecommendationConsumesExactPerEffectEvidence() && ok;
  ok = TestExplicitVulkanFilteringMatchesAudit() && ok;
  ok = TestEyeContactDiagnosticsOnlyContractIsFailClosed() && ok;
  ok = TestVideoNoiseRemovalDiagnosticsOnlyContractIsFailClosed() && ok;
  ok = TestMirrorProductionHelperIsCalledAtTheLiveFinalBoundary() && ok;
  ok = TestVignetteProductionHelperIsCalledAtTheLiveFinalBoundary() && ok;
  ok = TestVirtualBackgroundBlurUsesTheCanonicalResidentWrapper() && ok;
  ok = TestVirtualBackgroundRemoveUsesTheCanonicalResidentWrapper() && ok;
  ok = TestVirtualBackgroundReplaceUsesTheCanonicalResidentWrapper() && ok;
  ok = TestVirtualKeyLightUsesTheCanonicalResidentWrapper() && ok;
  ok = TestDefaultMattingDiagnosticsRemainFailClosed() && ok;
  ok = TestAutoFrameDegradedPathRemainsExplicit() && ok;
  ok = TestSharedVulkanSynchronizationAndLifetimeContracts() && ok;
  return ok ? 0 : 1;
}
