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
  bool ok = Require(
      input != std::string::npos && resize_binding != std::string::npos &&
          output_binding != std::string::npos &&
          mirror_binding != std::string::npos &&
          ordering_boundary != std::string::npos &&
          apply != std::string::npos && download != std::string::npos,
      "vignette live final-output helper call is incomplete");
  ok &=
      Require(input < resize_binding && resize_binding < output_binding &&
                  output_binding < mirror_binding &&
                  mirror_binding < ordering_boundary &&
                  ordering_boundary < apply && apply < download,
              "live path must batch output resize, vignette, optional mirror, "
              "then perform the final readback");
  ok &= Require(
      Contains(pipeline, "have_open_vulkan_vignette") &&
          Contains(pipeline,
                   "Open Vulkan: Vignette (fixed center after framing") &&
          Contains(pipeline,
                   "open_vulkan_vignette_counters.factor_generation_calls") &&
          Contains(pipeline,
                   "open_vulkan_vignette_counters.factor_upload_calls") &&
          Contains(pipeline, "set_backend(stage_id, \"open_vulkan\")"),
      "vignette backend, setup accounting, or fixed-center evidence is "
      "missing");

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
  ok &= Require(Contains(auto_frame, "CPU face tracking; Vulkan"),
                "Vulkan Auto Frame CPU face-tracking tail changed; audit "
                "required");
  ok &=
      Require(Contains(auto_frame, "CPU tracking tail and Vulkan"),
              "Vulkan Auto Frame matte tracking tail changed; audit required");
  ok &= Require(Contains(auto_frame, "set_backend(stage_id, \"open_vulkan\")"),
                "Vulkan Auto Frame is no longer connected to the live backend");
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
