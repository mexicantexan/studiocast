#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const char *message) {
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

std::string ReadPipelineSource() {
  const auto root = std::filesystem::path(STUDIOCAST_SOURCE_DIR);
  std::ifstream input(root / "src/core/video/camera_pipeline.cpp");
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

void TestNeutralPreparedVulkanRoute() {
  const std::string source = ReadPipelineSource();
  Check(!source.empty(), "camera pipeline source must be readable");
  if (source.empty())
    return;

  const auto setup = source.find("prepared_hardware_composition.Prepare");
  const auto loop = source.find("while (!stop_.load())");
  const auto execute =
      source.find("prepared_hardware_composition.Execute", loop);
  const auto write = source.find("writer_.WriteFrameDetailed", execute);
  const auto commit_gate =
      source.find("finish_prepared_hardware_frame(true)", write);
  Check(setup != std::string::npos && setup < loop,
        "Vulkan composition must prepare before the frame loop");
  Check(execute != std::string::npos && execute > loop,
        "the frame loop must execute through the neutral composition");
  Check(write != std::string::npos && commit_gate != std::string::npos &&
            write < commit_gate,
        "successful attribution commit must remain behind the writer");

  Check(source.find("PreparedHardwareExecutionOwner "
                    "prepared_hardware_owner") == std::string::npos,
        "the pipeline must not instantiate the retired Vulkan owner");
  Check(source.find("prepared_hardware_owner.ConfigureFinal") ==
            std::string::npos,
        "the pipeline must not retain the old Vulkan setup planner");
  Check(source.find("prepared_hardware_owner.ExecuteHostFrame") ==
            std::string::npos,
        "the pipeline must not retain the old Vulkan frame executor");

  const std::string live = source.substr(loop);
  Check(live.find("VulkanVideoExecutionFactory") == std::string::npos,
        "the frame loop must not instantiate or probe a Vulkan factory");
  Check(live.find("prepared_hardware_composition.Prepare") == std::string::npos,
        "the frame loop must not prepare or rebuild the Vulkan island");
}

void TestExactGatesAndFailureBoundaries() {
  const std::string source = ReadPipelineSource();
  Check(
      source.find("cfg.compute_backend == ComputeBackendPreference::vulkan") !=
          std::string::npos,
      "prepared Vulkan selection must remain explicit");
  Check(source.find("!contains_unqualified_stage") != std::string::npos,
        "unqualified stages must block the production Vulkan island");
  Check(source.find("fx.vignette.center_on_tracked_face") != std::string::npos,
        "tracked-center vignette must remain an exact fail-closed gate");
  Check(
      source.find("SemanticEffectProviderId::open_vulkan_production_kernels") !=
          std::string::npos,
      "the exact production Vulkan provider must be frozen");
  Check(source.find("VideoExecutionMigrationPolicy::prepared_only") !=
            std::string::npos,
        "the migrated route must not frame-fallback through compatibility");
  Check(source.find("setup.old_plan_retained") != std::string::npos,
        "setup rollback must retain only an exact old prepared plan");
  Check(source.find("prepared_frame.status == hal::StatusId::executed") !=
                std::string::npos &&
            source.find("prepared_frame.status == hal::StatusId::degraded") !=
                std::string::npos,
        "output acceptance must require successful neutral execution");
  Check(source.find("prepared_frame.output.output_ready") !=
                std::string::npos &&
            source.find("prepared_frame.output.planes[0].read_data") !=
                std::string::npos,
        "the writer must consume the exact resolved host output binding");
  Check(source.find("prepared_hardware_completion_pending = "
                    "prepared_output_valid") != std::string::npos,
        "only a validated neutral completion may reach the writer gate");
  Check(source.find("canonical_plan_hash = UINT64_C(1469598103934665603)") !=
            std::string::npos,
        "canonical invalidation must be semantic, not a changing plan id");
  Check(source.find("checked_multiply") != std::string::npos &&
            source.find("checked_add") != std::string::npos,
        "Vulkan resource budgeting must use checked setup-time arithmetic");
}

} // namespace

int main() {
  TestNeutralPreparedVulkanRoute();
  TestExactGatesAndFailureBoundaries();
  if (failures != 0) {
    std::cerr << failures
              << " production Vulkan pipeline migration checks failed\n";
    return 1;
  }
  std::cout << "production Vulkan pipeline migration checks passed\n";
  return 0;
}
