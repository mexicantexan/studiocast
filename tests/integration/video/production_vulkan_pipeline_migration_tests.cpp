#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

int failures = 0;

void Check(bool condition, const char *message) {
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

std::string ReadSource(std::string_view relative) {
  const auto root = std::filesystem::path(STUDIOCAST_SOURCE_DIR);
  std::ifstream input(root / relative);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

void TestPipelineUsesOnlyTheNeutralManager() {
  const std::string pipeline =
      ReadSource("src/core/video/camera_pipeline.cpp");
  Check(!pipeline.empty(), "camera pipeline source must be readable");
  if (pipeline.empty())
    return;

  const auto manager =
      pipeline.find("ProductionVideoBackendManager backend_manager");
  const auto setup = pipeline.find("backend_manager.Setup(setup_request");
  const auto loop = pipeline.find("while (!stop_.load())");
  const auto execute =
      pipeline.find("execution = backend_manager.Execute(binding)", loop);
  const auto write =
      pipeline.find("writer_.WriteFrameDetailed(write_data", execute);
  const auto commit =
      pipeline.find("backend_manager.CommitOutput(completion, written", write);
  Check(manager != std::string::npos && setup != std::string::npos &&
            loop != std::string::npos && execute != std::string::npos &&
            write != std::string::npos && commit != std::string::npos &&
            manager < setup && setup < loop && loop < execute &&
            execute < write && write < commit,
        "neutral manager setup/execute/writer-commit order changed");

  constexpr std::string_view forbidden[]{
      "VulkanVideoExecutionFactory",
      "PreparedHardwareExecutionOwner",
      "prepared_hardware_composition",
      "OpenVulkanFinalResidentStageInput",
      "DownloadImageToRgb",
  };
  for (const auto token : forbidden) {
    Check(pipeline.find(token) == std::string::npos,
          "camera pipeline regained Vulkan-specific orchestration");
  }
}

void TestVulkanSetupSourceOwnsExactSelectionAndTailBinding() {
  const std::string source = ReadSource(
      "src/core/vulkan/production_video_backend_setup_source.cpp");
  Check(!source.empty(), "Vulkan setup source must be readable");
  Check(source.find("caller.intent.backend.backend != "
                    "hal::BackendId::vulkan") != std::string::npos,
        "Vulkan setup source lost exact backend selection");
  Check(source.find("caller.intent.requested_provider != kProvider") !=
            std::string::npos,
        "Vulkan setup source lost exact provider selection");
  Check(source.find("semantic != hal::SemanticEffectId::mirror") !=
            std::string::npos &&
            source.find("semantic != hal::SemanticEffectId::vignette") !=
                std::string::npos,
        "Vulkan setup source must admit only mirror and vignette");
  Check(source.find("ExactTerminalBackendResolvedOutputTail") !=
            std::string::npos &&
            source.find("result.plan.cpu_tail_count = 0") !=
                std::string::npos,
        "Vulkan setup source must resolve the setup-only tail to zero");
  Check(source.find("result.plan.invalidation_key.execution_domain_hash") !=
            std::string::npos &&
            source.find("ProductionVideoSetupLease::Acquire") !=
                std::string::npos,
        "Vulkan setup source must bind a typed domain and owned lease");
}

void TestVulkanAdapterClaimsOnlyQualifiedOperations() {
  const std::string factory =
      ReadSource("src/core/vulkan/video_execution_factory.cpp");
  const std::string adapter =
      ReadSource("src/core/vulkan/hal_resident_adapter.cpp");
  Check(factory.find("semantic_capabilities[0] = ProductionCapability") !=
            std::string::npos &&
            factory.find("SemanticEffectId::vignette") !=
                std::string::npos &&
            factory.find("semantic_capabilities[1] = ProductionCapability") !=
                std::string::npos &&
            factory.find("SemanticEffectId::mirror") != std::string::npos &&
            factory.find("semantic_capability_count = 2") !=
                std::string::npos,
        "Vulkan factory capability set must remain exactly two effects");
  Check(adapter.find("program.cpu_tail_count != 0") != std::string::npos &&
            adapter.find("ReasonId::cpu_tail_required") !=
                std::string::npos &&
            adapter.find("ReasonId::semantic_effect_unsupported") !=
                std::string::npos,
        "Vulkan adapter lost zero-tail or unsupported-effect gates");
  Check(factory.find("vulkan_exact_device_shared") != std::string::npos &&
            factory.find("vulkan_no_intermediate_readbacks") !=
                std::string::npos,
        "Vulkan factory lost exact-domain or residency evidence");
}

} // namespace

int main() {
  TestPipelineUsesOnlyTheNeutralManager();
  TestVulkanSetupSourceOwnsExactSelectionAndTailBinding();
  TestVulkanAdapterClaimsOnlyQualifiedOperations();
  if (failures != 0) {
    std::cerr << failures
              << " production Vulkan pipeline migration checks failed\n";
    return 1;
  }
  std::cout << "production Vulkan pipeline migration checks passed\n";
  return 0;
}
