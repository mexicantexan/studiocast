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
    {kEffectIdMirror, "stub/unavailable", "vulkan_effect_not_implemented"},
    {kEffectIdVirtualBackgroundBlur, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdVirtualBackgroundRemove, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdVirtualBackgroundReplace, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdAutoFrame, "usable with degraded behavior",
     "vulkan_effect_cpu_tail"},
    {kEffectIdEyeContact, "stub/unavailable", "vulkan_effect_not_implemented"},
    {kEffectIdVideoNoiseRemoval, "stub/unavailable",
     "vulkan_effect_not_implemented"},
    {kEffectIdVirtualKeyLight, "experimental",
     "open_vulkan_matting_unavailable"},
    {kEffectIdVignette, "stub/unavailable", "vulkan_effect_not_implemented"},
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
  ok &= Require(Contains(audit, "No canonical effect is currently in the "
                                "**production usable** Vulkan category."),
                "audit must retain the conservative recommendation gate");
  return ok;
}

bool TestExplicitVulkanFilteringMatchesAudit() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::string filtering =
      Section(pipeline, "remove_non_vulkan_vb_compute_stages_from_plan",
              "const bool vb_requested");
  bool ok = Require(!filtering.empty(),
                    "explicit Vulkan filtering section not found");
  ok &= Require(Contains(filtering, "kEffectIdVideoNoiseRemoval"),
                "explicit Vulkan must continue removing video denoise");
  ok &= Require(Contains(filtering, "kEffectIdEyeContact"),
                "explicit Vulkan must continue removing eye contact");
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

bool TestVignetteProductionHelperIsCalledAtTheLiveFinalBoundary() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::size_t input =
      pipeline.find("OpenVulkanVignetteFinalStageInput vignette_input");
  const std::size_t resize_binding =
      pipeline.find("vignette_input.resize_scratch", input);
  const std::size_t output_binding =
      pipeline.find("vignette_input.dst", resize_binding);
  const std::size_t mirror_binding =
      pipeline.find("vignette_input.mirrored_dst", output_binding);
  const std::size_t ordering_boundary = pipeline.find(
      "vignette_input.preceding_effects_complete = true", mirror_binding);
  const std::size_t apply =
      pipeline.find("open_vulkan_vignette.ApplyFinal", ordering_boundary);
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
      input != std::string::npos && resize_binding != std::string::npos &&
          output_binding != std::string::npos &&
          mirror_binding != std::string::npos &&
          ordering_boundary != std::string::npos &&
          apply != std::string::npos && download != std::string::npos &&
          auto_frame_setup != std::string::npos &&
          auto_frame_resolution != std::string::npos &&
          compatibility_call != std::string::npos &&
          vignette_setup != std::string::npos &&
          compatibility_status_publish != std::string::npos,
      "vignette live final-output helper call is incomplete");
  ok &= Require(
      input < resize_binding && resize_binding < output_binding &&
          output_binding < mirror_binding &&
          mirror_binding < ordering_boundary && ordering_boundary < apply &&
          apply < download && auto_frame_setup < auto_frame_resolution &&
          auto_frame_resolution < compatibility_call &&
          compatibility_call < vignette_setup &&
          vignette_setup < compatibility_status_publish,
      "live path must batch output resize, vignette, optional mirror, "
      "then perform the final readback, with compatibility evaluated "
      "after Auto Frame setup and its stable block published to status");
  ok &= Require(
      Contains(pipeline, "have_open_vulkan_vignette") &&
          Contains(pipeline,
                   "Open Vulkan: Vignette (fixed center after framing") &&
          Contains(pipeline, "device-resident attenuation mask") &&
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
  ok &= Require(Contains(helper, "OpenVulkanVignetteRuntimeFailure") &&
                    Contains(helper, "ApplyFinalVignetteU8x3") &&
                    Contains(helper, "production_hardware_ready") &&
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
  const std::size_t combined_input =
      pipeline.find("OpenVulkanMirrorResizeFinalStageInput mirror_input");
  const std::size_t analysis_boundary = pipeline.find(
      "mirror_input.unmirrored_analysis_complete = true", combined_input);
  const std::size_t combined_apply =
      pipeline.find("open_vulkan_mirror.ApplyResizeFinal", analysis_boundary);
  const std::size_t geometry_boundary = pipeline.find(
      "mirror_input.output_geometry_ready = true", combined_apply);
  const std::size_t apply =
      pipeline.find("open_vulkan_mirror.ApplyFinal", geometry_boundary);
  const std::size_t download =
      pipeline.find("DownloadImageToRgb(*download_img", apply);
  bool ok =
      Require(combined_input != std::string::npos &&
                  analysis_boundary != std::string::npos &&
                  combined_apply != std::string::npos &&
                  geometry_boundary != std::string::npos &&
                  apply != std::string::npos && download != std::string::npos,
              "mirror live final-output helper call is incomplete");
  ok &= Require(combined_input < analysis_boundary &&
                    analysis_boundary < combined_apply &&
                    combined_apply < geometry_boundary &&
                    geometry_boundary < apply && apply < download,
                "live path must use the combined resize/mirror helper, retain "
                "the same-size final helper, and mirror before readback");
  ok &= Require(
      Contains(pipeline, "have_open_vulkan_mirror") &&
          Contains(pipeline, "open_vulkan_mirror.ApplyFinal") &&
          Contains(pipeline, "open_vulkan_mirror.ApplyResizeFinal") &&
          Contains(pipeline, "set_backend(stage_id, \"open_vulkan\")") &&
          Contains(pipeline, "open_vulkan_mirror_counters.dispatch_calls"),
      "mirror batching, applied-backend, or counter evidence is missing");

  const std::string helper =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_mirror.cpp");
  ok &= Require(Contains(helper, "vulkan_effect_initialization_failed") ||
                    Contains(ReadFile(root / "src" / "core" / "video" /
                                      "open_vulkan_mirror.h"),
                             "vulkan_effect_initialization_failed"),
                "mirror initialization stable reason is missing");
  ok &= Require(Contains(helper, "OpenVulkanMirrorRuntimeFailure") &&
                    Contains(helper, "MirrorHorizontalU8x3") &&
                    Contains(helper, "ResizeMirrorHorizontalU8x3") &&
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
                    Contains(blocker, "kEffectIdVirtualBackgroundReplace"),
                "matting blocker must cover all virtual-background modes");
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
          Contains(live_context, "production_crop.ApplyCrop") &&
          !Contains(live_context, ".CropResizeBilinear"),
      "the live Auto Frame context must use the reset/reuse contract and route "
      "crop/resize only through the canonical production wrapper");

  const std::size_t cache_begin =
      pipeline.find("open_video_cache.BeginFrame(capture_sequence)");
  const std::size_t preupload_face =
      pipeline.find("YunetProviderPolicy::cpu_only", cache_begin);
  const std::size_t upload = pipeline.find(
      "ensure_open_vulkan_current_from_cpu(&vk_err)", preupload_face);
  const std::size_t canonical_crop = pipeline.find("production_crop.ApplyCrop");
  ok &= Require(
      cache_begin != std::string::npos && preupload_face != std::string::npos &&
          upload != std::string::npos && canonical_crop != std::string::npos &&
          cache_begin < preupload_face && preupload_face < upload,
      "Auto Frame must analyze the original host capture before "
      "the one Vulkan upload and use the canonical crop wrapper");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = TestAuditCoversCanonicalEffects() && ok;
  ok = TestExplicitVulkanFilteringMatchesAudit() && ok;
  ok = TestMirrorProductionHelperIsCalledAtTheLiveFinalBoundary() && ok;
  ok = TestVignetteProductionHelperIsCalledAtTheLiveFinalBoundary() && ok;
  ok = TestDefaultMattingDiagnosticsRemainFailClosed() && ok;
  ok = TestAutoFrameDegradedPathRemainsExplicit() && ok;
  return ok ? 0 : 1;
}
