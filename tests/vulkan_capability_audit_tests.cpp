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
  ok &= Require(Contains(filtering, "kEffectIdVignette"),
                "explicit Vulkan must continue removing vignette");
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

bool TestMirrorProductionHelperIsCalledAtTheLiveFinalBoundary() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src" / "core" / "video" / "camera_pipeline.cpp");
  const std::size_t resize =
      pipeline.find("deferred_gpu_out.vulkan_kernels->ResizeBilinear");
  const std::size_t analysis_boundary =
      pipeline.find("mirror_input.unmirrored_analysis_complete = true");
  const std::size_t geometry_boundary =
      pipeline.find("mirror_input.output_geometry_ready = true");
  const std::size_t apply = pipeline.find("open_vulkan_mirror.ApplyFinal");
  const std::size_t download =
      pipeline.find("DownloadImageToRgb(*download_img", apply);
  bool ok = Require(
      resize != std::string::npos && analysis_boundary != std::string::npos &&
          geometry_boundary != std::string::npos &&
          apply != std::string::npos && download != std::string::npos,
      "mirror live final-output helper call is incomplete");
  ok &= Require(resize < analysis_boundary && analysis_boundary < apply &&
                    geometry_boundary < apply && apply < download,
                "live path must resize before mirror and mirror before final "
                "readback");
  ok &= Require(
      Contains(pipeline, "have_open_vulkan_mirror") &&
          Contains(pipeline, "set_backend(stage_id, \"open_vulkan\")") &&
          Contains(pipeline, "open_vulkan_mirror_counters.dispatch_calls"),
      "mirror applied-backend and counter evidence is missing");

  const std::string helper =
      ReadFile(root / "src" / "core" / "video" / "open_vulkan_mirror.cpp");
  ok &= Require(Contains(helper, "vulkan_effect_initialization_failed") ||
                    Contains(ReadFile(root / "src" / "core" / "video" /
                                      "open_vulkan_mirror.h"),
                             "vulkan_effect_initialization_failed"),
                "mirror initialization stable reason is missing");
  ok &= Require(Contains(helper, "OpenVulkanMirrorRuntimeFailure") &&
                    Contains(helper, "MirrorHorizontalU8x3"),
                "mirror runtime stable reason or real kernel call is missing");
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
  ok = TestDefaultMattingDiagnosticsRemainFailClosed() && ok;
  ok = TestAutoFrameDegradedPathRemainsExplicit() && ok;
  return ok ? 0 : 1;
}
