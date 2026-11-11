#include "core/video/camera_pipeline.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR "."
#endif

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": "            \
                << #condition << '\n';                                         \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

std::string ReadSource(
    std::string_view relative_path = "src/core/video/camera_pipeline.cpp") {
  const auto root = std::filesystem::path(STUDIOCAST_SOURCE_DIR);
  std::ifstream input(root / relative_path);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

std::size_t Count(std::string_view text, std::string_view needle) {
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string_view::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

void TestNeutralManagerOwnsThePipeline() {
  const std::string source = ReadSource();
  CHECK(!source.empty());
  if (source.empty())
    return;

  const auto manager =
      source.find("ProductionVideoBackendManager backend_manager");
  const auto setup_request = source.find("auto setup_request = BuildSetupRequest");
  const auto setup = source.find("backend_manager.Setup(setup_request");
  const auto loop = source.find("while (!stop_.load())");
  const auto execute = source.find("execution = backend_manager.Execute(binding)");
  const auto write = source.find("writer_.WriteFrameDetailed(write_data");
  const auto commit = source.find("backend_manager.CommitOutput(completion, written");
  const auto shutdown =
      source.find("backend_manager.Invalidate(hal::ReasonId::shutdown_requested)");

  CHECK(manager != std::string::npos);
  CHECK(setup_request != std::string::npos);
  CHECK(setup != std::string::npos);
  CHECK(loop != std::string::npos);
  CHECK(execute != std::string::npos);
  CHECK(write != std::string::npos);
  CHECK(commit != std::string::npos);
  CHECK(shutdown != std::string::npos);
  CHECK(manager < setup_request && setup_request < setup && setup < loop);
  CHECK(loop < execute && execute < write && write < commit &&
        commit < shutdown);
  CHECK(Count(source, "backend_manager.Execute(binding)") == 1);
  CHECK(source.find("CreateStaticProductionVideoBackendSetupSource") !=
        std::string::npos);
  CHECK(source.find("BuildProductionVideoBackendSetupRequest(") !=
        std::string::npos);
  const auto builder =
      ReadSource("src/core/video/production_video_setup_request.cpp");
  CHECK(!builder.empty());
  CHECK(builder.find("hal::CpuTailId::backend_resolved_output") !=
        std::string::npos);
}

void TestBackendConstructionAndDiscoveryStayOutOfThePipeline() {
  const std::string source = ReadSource();
  CHECK(!source.empty());
  if (source.empty())
    return;

  constexpr std::string_view forbidden[]{
      "CreateMaxineProductionVideoBackendSetupSource",
      "open_cuda::CreateProductionVideoBackendSetupSource",
      "vulkan::CreateProductionVideoBackendSetupSource",
      "MaxineManager",
      "OpenCudaVideoExecutionFactory",
      "VulkanVideoExecutionFactory",
      "ModelPackRegistry",
      "ValidateModelArtifactIdentity",
      "AcquireProductionSetupLease",
      "CudaImage",
      "CudaTensor",
      "NvcvApi",
      "VfxApi",
      "ArApi",
  };
  for (const auto token : forbidden)
    CHECK(source.find(token) == std::string::npos);

  constexpr std::string_view retired_orchestration[]{
      "PreparedOpenCudaHardwareExecutionOwner",
      "PreparedMaxineHardwareExecutionOwner",
      "PreparedHardwareExecutionOwner",
      "open_cuda_hardware_composition",
      "maxine_hardware_composition",
      "vulkan_hardware_composition",
      "try_prepare_open_cuda_virtual_background",
  };
  for (const auto token : retired_orchestration)
    CHECK(source.find(token) == std::string::npos);
}

void TestFrameLoopHasNoDiscoveryOrSerialization() {
  const std::string source = ReadSource();
  const auto loop = source.find("while (!stop_.load())");
  CHECK(loop != std::string::npos);
  if (loop == std::string::npos)
    return;
  const std::string_view live(source.data() + loop, source.size() - loop);

  constexpr std::string_view forbidden[]{
      "ScanDefault",
      "ValidateModelArtifactIdentity",
      "LoadImageRgb24",
      "dlopen",
      "getenv",
      "Json",
      "JSON",
      ".Realloc(",
  };
  for (const auto token : forbidden)
    CHECK(live.find(token) == std::string_view::npos);
}

} // namespace

int main() {
  TestNeutralManagerOwnsThePipeline();
  TestBackendConstructionAndDiscoveryStayOutOfThePipeline();
  TestFrameLoopHasNoDiscoveryOrSerialization();
  if (failures != 0) {
    std::cerr << failures
              << " neutral pipeline migration checks failed\n";
    return 1;
  }
  std::cout << "neutral pipeline migration checks passed\n";
  return 0;
}
