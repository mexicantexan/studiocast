#include "core/video/effects/broadcast_effect_contract.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

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
    {kEffectIdVirtualBackgroundBlur, "production usable (opt-in matting)",
     "open_vulkan_matting_production_ready"},
    {kEffectIdVirtualBackgroundRemove, "production usable (opt-in matting)",
     "open_vulkan_matting_production_ready"},
    {kEffectIdVirtualBackgroundReplace, "production usable (opt-in matting)",
     "open_vulkan_matting_production_ready"},
    {kEffectIdAutoFrame,
     "resident implementation; exact qualification required",
     "vulkan_auto_frame_yunet_unavailable"},
    {kEffectIdEyeContact, "resident implementation; exact qualification required",
     "open_vulkan_eye_contact_unavailable"},
    {kEffectIdVideoNoiseRemoval, "resident implementation; exact qualification required",
     "open_vulkan_video_noise_removal_unavailable"},
    {kEffectIdVirtualKeyLight, "production usable (opt-in matting)",
     "open_vulkan_matting_production_ready"},
    {kEffectIdVignette, "fixed and tracked resident implementations",
     "open_vulkan_vignette_fixed_center_production_ready"},
}};

std::string ReadFile(const fs::path &path) {
  std::ifstream input(path);
  if (!input)
    return {};
  std::ostringstream contents;
  contents << input.rdbuf();
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
    const std::string prefix = "| `" + std::string(row.effect_id) + "` | " +
                               std::string(row.status) + " |";
    ok &= Require(Contains(audit, prefix),
                  "audit row/status missing for " + std::string(row.effect_id));
    ok &= Require(Contains(audit, "`" + std::string(row.reason_code) + "`"),
                  "audit reason missing for " + std::string(row.effect_id));
  }
  ok &= Require(ids.size() == kExpectedCapabilities.size(),
                "capability matrix must cover exactly nine effects");
  ok &= Require(
      Contains(audit, "The four matting consumers additionally require") &&
          Contains(audit, "matting_parity_validated") &&
          Contains(audit, "`effect.<id>.cpu.no_selectable_production_path`") &&
          Contains(audit, "`available_effects` is useful live-stage evidence "
                          "but is never sufficient"),
      "audit must retain exact effect and CPU recommendation gates");
  return ok;
}

bool TestInstallerConsumesExactPerEffectEvidence() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string facts =
      ReadFile(root / "installer" / "backend" /
               "studiocast_installer_backend" / "facts.py");
  const std::string recommendation =
      ReadFile(root / "installer" / "backend" /
               "studiocast_installer_backend" / "recommendation.py");
  bool ok = Require(!facts.empty() && !recommendation.empty(),
                    "installer Vulkan evidence sources are missing");
  ok &= Require(
      Contains(facts, "def common_vulkan_effect_evidence") &&
          Contains(facts, "runtime_diagnostics_available") &&
          Contains(facts, "non_cpu_device_selected") &&
          Contains(facts, "compute_queue_available") &&
          Contains(facts, "context_healthy") &&
          Contains(facts, "utility_kernels_ready") &&
          Contains(recommendation, "def engine_capability_usable") &&
          Contains(recommendation,
                   "all(evidence.get(key) is True for key in common_keys)") &&
          Contains(recommendation, "expected_success in successes.get"),
      "recommendation must require common and exact Vulkan proof");
  ok &= Require(
      Contains(facts, "mirror_production_ready") &&
          Contains(facts, "vignette_fixed_center_production_ready") &&
          Contains(facts, "auto_frame_cpu_tail") &&
          Contains(facts, "success_reason_codes") &&
          Contains(facts, "blocker_reason_codes") &&
          Contains(facts, "selectable_production_path") &&
          Contains(facts,
                   "effect.{effect}.cpu.no_selectable_production_path") &&
          Contains(recommendation, "success_reason_codes") &&
          Contains(recommendation, "blocker_reason_codes") &&
          Contains(recommendation,
                   "cpu_evidence.get(\"selectable_production_path\") is True"),
      "per-effect success, blocker, and CPU facts are incomplete");
  return ok;
}

bool TestStaticHalPreservesTheExactProductionSet() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string pipeline =
      ReadFile(root / "src/core/video/camera_pipeline.cpp");
  const std::string source = ReadFile(
      root / "src/core/vulkan/production_video_backend_setup_source.cpp");
  const std::string factory =
      ReadFile(root / "src/core/vulkan/video_execution_factory.cpp");
  const std::string adapter =
      ReadFile(root / "src/core/vulkan/hal_resident_adapter.cpp");
  bool ok = Require(!pipeline.empty() && !source.empty() && !factory.empty() &&
                        !adapter.empty(),
                    "static Vulkan HAL sources are missing");
  ok &= Require(
      !Contains(pipeline, "VulkanVideoExecutionFactory") &&
          !Contains(pipeline, "PreparedHardwareExecutionOwner") &&
          Contains(pipeline, "ProductionVideoBackendManager backend_manager"),
      "camera pipeline must not regain Vulkan-specific orchestration");
  ok &= Require(
      Contains(source, "caller.intent.backend.backend != "
                       "hal::BackendId::vulkan") &&
          Contains(source, "caller.intent.requested_provider != kProvider") &&
          Contains(source, "semantic != hal::SemanticEffectId::mirror") &&
          Contains(source, "semantic != hal::SemanticEffectId::vignette") &&
          Contains(source, "ValidateProductionNcnnVulkanMattingPack") &&
          Contains(source, "snapshot.artifacts") &&
          Contains(source, "ExactTerminalBackendResolvedOutputTail") &&
          Contains(source, "result.plan.cpu_tail_count = 0"),
      "Vulkan setup must bind strict selection, verified model artifacts and "
      "zero CPU tails");
  ok &= Require(
      Contains(factory, "semantic_capabilities[0] = ProductionCapability") &&
          Contains(factory, "SemanticEffectId::vignette") &&
          Contains(factory, "semantic_capabilities[1] = ProductionCapability") &&
          Contains(factory, "SemanticEffectId::mirror") &&
          Contains(factory, "semantic_capability_count = 2") &&
          Contains(factory, "matting_production_ready") &&
          Contains(factory, "vulkan_model_artifacts_valid") &&
          Contains(factory, "SemanticEffectId::virtual_background_blur") &&
          Contains(factory, "SemanticEffectId::virtual_background_remove") &&
          Contains(factory, "SemanticEffectId::virtual_background_replace") &&
          Contains(factory, "SemanticEffectId::virtual_key_light") &&
          Contains(factory, "vulkan_exact_device_shared") &&
          Contains(factory, "vulkan_no_intermediate_readbacks"),
      "Vulkan factory capability set or evidence changed");
  ok &= Require(
      Contains(adapter, "program.cpu_tail_count != 0") &&
          Contains(adapter, "ReasonId::cpu_tail_required") &&
          Contains(adapter, "ReasonId::semantic_effect_unsupported") &&
          Contains(adapter, "exact_device_and_queue_shared") &&
          Contains(adapter, "cross_adapter_residency_forbidden"),
      "Vulkan adapter fail-closed domain/effect gates changed");
  return ok;
}

bool TestBlockedAndExperimentalEvidenceRemainsFailClosed() {
  const fs::path root(STUDIOCAST_SOURCE_DIR);
  const std::string diagnostics =
      ReadFile(root / "src/core/vulkan/kernels/resize_bilinear.cpp");
  const std::string device =
      ReadFile(root / "src/core/vulkan/vulkan_device.h");
  const std::string eye = ReadFile(
      root / "src/core/video/open_vulkan_eye_contact.h");
  const std::string denoise = ReadFile(
      root / "src/core/video/open_vulkan_video_noise_removal.h");
  const std::string spike =
      ReadFile(root / "docs/vulkan_matting_ncnn_spike.md");
  const std::string native = ReadFile(
      root / "src/core/open_video/ncnn_vulkan_matting_runtime.cpp");
  const std::string dependency = ReadFile(root / "cmake/Ncnn.cmake");
  bool ok = Require(!diagnostics.empty() && !device.empty() && !eye.empty() &&
                        !denoise.empty() && !spike.empty(),
                    "Vulkan fail-closed evidence sources are missing");
  ok &= Require(
      Contains(diagnostics, "ApplyOpenVulkanAutoFrameReleaseReadiness") &&
          Contains(diagnostics, "ApplyOpenVulkanEyeContactReadiness") &&
          Contains(diagnostics,
                   "ApplyOpenVulkanVideoNoiseRemovalReadiness") &&
          Contains(device, "auto_frame_production_ready = false") &&
          Contains(device, "eye_contact_production_ready = false") &&
          Contains(device, "video_noise_removal_production_ready = false") &&
          Contains(device, "matting_production_ready = false"),
      "blocked Vulkan readiness defaults must remain false");
  ok &= Require(
      Contains(eye, "live_stage_implemented = false") &&
          Contains(eye, "production_adapter_available = false") &&
          Contains(eye, "selectable_cpu_fallback = false") &&
          Contains(denoise, "live_stage_implemented = false") &&
          Contains(denoise, "production_adapter_available = false") &&
          Contains(denoise, "selectable_cpu_fallback = false"),
      "model readiness defaults must not infer live adapter or selectable CPU fallback");
  ok &= Require(
      Contains(spike, "not a production capability contract") &&
          Contains(native, "OpenVerifiedModelFileSnapshot") &&
          Contains(native, "net.load_param(parameters.get())") &&
          Contains(native, "net.load_model(weights.get())") &&
          Contains(dependency, "NCNN_STUDIOCAST_REPLAY_ABI != 2"),
      "production matting needs verified snapshots and the bounded replay ABI; "
      "the historical spike cannot attest readiness");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = TestAuditCoversCanonicalEffects() && ok;
  ok = TestInstallerConsumesExactPerEffectEvidence() && ok;
  ok = TestStaticHalPreservesTheExactProductionSet() && ok;
  ok = TestBlockedAndExperimentalEvidenceRemainsFailClosed() && ok;
  return ok ? 0 : 1;
}
