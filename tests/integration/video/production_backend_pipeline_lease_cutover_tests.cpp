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

void TestSeparateNeutralIslandsOwnBackendLifetimes() {
  const std::string source = ReadPipelineSource();
  Check(!source.empty(), "camera pipeline source must be readable");
  if (source.empty())
    return;

  Check(source.find(
            "ProductionVideoExecutionComposition "
            "prepared_hardware_composition") != std::string::npos,
        "Vulkan final execution must keep its own neutral composition");
  Check(source.find(
            "ProductionVideoExecutionComposition "
            "maxine_hardware_composition") != std::string::npos,
        "Maxine execution must use a distinct neutral composition");
  Check(source.find(
            "ProductionVideoExecutionComposition "
            "open_cuda_hardware_composition") != std::string::npos,
        "Open CUDA execution must use a distinct neutral composition");

  Check(source.find("PreparedMaxineHardwareExecutionOwner "
                    "prepared_maxine_hardware_owner") ==
            std::string::npos,
        "CameraPipeline must not construct the Maxine prepared owner");
  Check(source.find("PreparedOpenCudaHardwareExecutionOwner "
                    "prepared_open_cuda_hardware_owner") ==
            std::string::npos,
        "CameraPipeline must not construct the Open CUDA prepared owner");
  Check(source.find("prepared_maxine_hardware_owner.") ==
            std::string::npos,
        "CameraPipeline must not call the Maxine prepared owner");
  Check(source.find("prepared_open_cuda_hardware_owner.") ==
            std::string::npos,
        "CameraPipeline must not call the Open CUDA prepared owner");

  Check(source.find("AcquireSetupLease") != std::string::npos,
        "Maxine native setup must enter through a typed setup lease");
  Check(source.find("AcquireProductionSetupLease") !=
            std::string::npos,
        "Open CUDA native setup must enter through a typed setup lease");
  Check(source.find(
            "maxine_hardware_composition.PrepareWithSetupLease") !=
            std::string::npos &&
            source.find(
                "open_cuda_hardware_composition.PrepareWithSetupLease") !=
                std::string::npos,
        "both backend leases must be prepared by neutral coordinators");
  Check(source.find("mixed_providers") != std::string::npos &&
            source.find(
                "intent.strict_provider_selection =\n"
                "              !mixed_providers") != std::string::npos,
        "mixed Maxine providers must remain exact per stage");
}

void TestPreparedFrameAndWriterBoundaries() {
  const std::string source = ReadPipelineSource();
  const auto loop = source.find("while (!stop_.load())");
  const auto maxine_execute =
      source.find("maxine_hardware_composition.Execute(binding)", loop);
  const auto open_cuda_execute =
      source.find("open_cuda_hardware_composition.Execute(binding)", loop);
  const auto write =
      source.find("writer_.WriteFrameDetailed", open_cuda_execute);
  Check(loop != std::string::npos && maxine_execute != std::string::npos &&
            maxine_execute > loop,
        "Maxine frames must execute through the neutral composition");
  Check(open_cuda_execute != std::string::npos &&
            open_cuda_execute > loop,
        "Open CUDA frames must execute through the neutral composition");
  Check(write != std::string::npos,
        "prepared completion must remain coupled to the writer");
  Check(source.find("maxine_hardware_composition.CommitOutput") !=
            std::string::npos &&
            source.find("open_cuda_hardware_composition.CommitOutput") !=
                std::string::npos &&
            source.find("finish_prepared_hardware_frame(true)", write) !=
                std::string::npos,
        "successful backend attribution must commit after the writer");

  Check(source.find("frame.transfers.uploads == 1") !=
            std::string::npos &&
            source.find("frame.transfers.final_readbacks == 1") !=
                std::string::npos &&
            source.find("frame.transfers.intermediate_uploads == 0") !=
                std::string::npos &&
            source.find("frame.transfers.intermediate_readbacks == 0") !=
                std::string::npos,
        "Maxine output must enforce one upload/readback and no crossings");
  Check(source.find("prepared_frame.transfers.uploads == 1") !=
            std::string::npos &&
            source.find(
                "prepared_frame.transfers.final_readbacks == 1") !=
                std::string::npos &&
            source.find(
                "prepared_frame.transfers.intermediate_uploads == 0") !=
                std::string::npos &&
            source.find(
                "prepared_frame.transfers.intermediate_readbacks == 0") !=
                std::string::npos,
        "Open CUDA output must enforce one upload/readback and no crossings");
  Check(source.find(
            "VideoExecutionMigrationPolicy::prepared_only") !=
            std::string::npos &&
            source.find(
                "hal::SetupFailurePolicy::retain_old_program") !=
                std::string::npos &&
            source.find("exact_old_retained") != std::string::npos,
        "cutover must be prepared-only with exact old-plan rollback");
  Check(source.find(
            "Never retry through the legacy context") !=
            std::string::npos,
        "prepared Open CUDA failure must not retry compatibility in-frame");
}

void TestStatusAndShutdownRemainBounded() {
  const std::string source = ReadPipelineSource();
  Check(source.find("prepared_maxine_snapshot") != std::string::npos &&
            source.find("prepared_open_cuda_snapshot") !=
                std::string::npos &&
            source.find(
                "projection.maxine_resident = "
                "prepared_maxine_snapshot()") != std::string::npos &&
            source.find(
                "projection.open_cuda_virtual_background =\n"
                "            prepared_open_cuda_snapshot()") !=
                std::string::npos,
        "compatibility status must project neutral coordinator snapshots");
  Check(source.find(
            "open_cuda_hardware_composition.ShutdownSnapshot().pending") !=
            std::string::npos &&
            source.find(
                "maxine_hardware_composition.ShutdownSnapshot().pending") !=
                std::string::npos,
        "shutdown retries must be conditional on typed pending evidence");
  Check(source.find("attempt < 8") != std::string::npos,
        "shutdown retries must remain explicitly bounded");
  Check(source.find(
            "shutdown.quiescent && "
            "shutdown.driver_context_release_permitted") !=
            std::string::npos,
        "Open CUDA stopped status must require proven quiescence");
}

} // namespace

int main() {
  TestSeparateNeutralIslandsOwnBackendLifetimes();
  TestPreparedFrameAndWriterBoundaries();
  TestStatusAndShutdownRemainBounded();
  if (failures != 0) {
    std::cerr << failures
              << " production backend lease cutover checks failed\n";
    return 1;
  }
  std::cout << "production backend lease cutover checks passed\n";
  return 0;
}
