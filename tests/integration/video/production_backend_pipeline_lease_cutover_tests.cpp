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

std::string Read(const char *relative) {
  const auto root = std::filesystem::path(STUDIOCAST_SOURCE_DIR);
  std::ifstream input(root / relative);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

void TestPublicHeaderIsBackendNeutral() {
  const auto header = Read("src/core/video/camera_pipeline.h");
  Check(!header.empty(), "camera pipeline header must be readable");
  if (header.empty())
    return;

  constexpr const char *forbidden[] = {
      "#include \"core/maxine/",
      "#include \"core/open_video/",
      "#include \"core/cuda/",
      "#include \"core/vulkan/",
      "prepared_hardware_execution_owner.h",
      "prepared_maxine_hardware_execution_owner.h",
      "prepared_open_cuda_hardware_execution_owner.h",
      "maxine::",
      "PreparedMaxineHardwareExecutionSnapshot",
      "PreparedOpenCudaHardwareExecutionSnapshot",
      "PreparedHardwareExecutionSnapshot",
      "#include \"core/video/open_vulkan_eye_contact.h\"",
      "#include \"core/video/open_vulkan_video_noise_removal.h\"",
  };
  for (const auto *needle : forbidden)
    Check(header.find(needle) == std::string::npos,
          "public header must not expose an optional backend type");

  Check(header.find("#include \"core/video/"
                    "production_video_backend_manager.h\"") !=
            std::string::npos,
        "public header must consume the neutral backend manager");
  Check(header.find("ProductionVideoBackendSetupSourceFactory") !=
            std::string::npos &&
            header.find("backend_setup_source_factory") != std::string::npos,
        "tests must have a neutral setup-source injection seam");
  Check(header.find("ProductionVideoBackendManagerSnapshot "
                    "production_backend") != std::string::npos,
        "status must publish the neutral manager snapshot");
}

void TestPipelineOwnsNoBackendOrCompatibilityPlanner() {
  const auto source = Read("src/core/video/camera_pipeline.cpp");
  Check(!source.empty(), "camera pipeline source must be readable");
  if (source.empty())
    return;

  constexpr const char *forbidden[] = {
      "#include \"core/maxine/",
      "#include \"core/open_video/",
      "#include \"core/cuda/",
      "#include \"core/vulkan/",
      "PreparedMaxineHardwareExecutionOwner",
      "PreparedOpenCudaHardwareExecutionOwner",
      "PreparedHardwareExecutionOwner",
      "ProductionVideoExecutionComposition",
      "AcquireSetupLease",
      "AcquireProductionSetupLease",
      "EnsureInitialized(",
      "ReallocIfNeeded",
      "NvCVImage_Realloc",
      "ModelPackRegistry",
      ".Diagnose(",
      "OpenCudaMattingSession",
      "OpenVulkanMattingSession",
      "maxine_hardware_composition",
      "open_cuda_hardware_composition",
      "prepared_hardware_composition",
  };
  for (const auto *needle : forbidden)
    Check(source.find(needle) == std::string::npos,
          "pipeline must not retain backend setup or frame orchestration");

  Check(source.find("CreateStaticProductionVideoBackendSetupSource()") !=
            std::string::npos,
        "production must obtain one neutral static setup source");
  Check(source.find("ProductionVideoBackendManager backend_manager") !=
            std::string::npos,
        "pipeline must own exactly one neutral backend manager");
  Check(source.find("BuildSetupRequest(") != std::string::npos &&
            source.find("ProductionVideoBackendSetupRequest request") !=
                std::string::npos,
        "setup must originate in one canonical neutral request");
}

void TestFrameLoopUsesOnlyPreparedManagerLifecycle() {
  const auto source = Read("src/core/video/camera_pipeline.cpp");
  const auto loop = source.find("while (!stop_.load())");
  const auto acquire = source.find("capture.AcquireFrame", loop);
  const auto execute = source.find("backend_manager.Execute(binding)", acquire);
  const auto write =
      source.find("writer_.WriteFrameDetailed", execute);
  const auto commit =
      source.find("backend_manager.CommitOutput(completion, written", write);
  Check(loop != std::string::npos && acquire != std::string::npos,
        "prepared live loop must acquire frames");
  Check(execute != std::string::npos && execute > acquire,
        "semantic execution must call only the neutral manager");
  Check(write != std::string::npos && commit != std::string::npos &&
            execute < write && write < commit,
        "writer-gated attribution must commit only after output write");

  const auto frame_region = source.substr(acquire, write - acquire);
  constexpr const char *frame_forbidden[] = {
      "BuildBroadcastEffectsPlan(",
      "CreateStaticProductionVideoBackendSetupSource(",
      "backend_manager.Setup(",
      "ModelPack",
      "Diagnose",
      "EnsureInitialized",
      "Realloc",
  };
  for (const auto *needle : frame_forbidden)
    Check(frame_region.find(needle) == std::string::npos,
          "ordinary frame execution must not probe, plan, or allocate");

  Check(source.find("attempted_generation") != std::string::npos &&
            source.find("published_generation, attempted_generation") !=
                std::string::npos,
        "failed setup generation must be cached instead of retried per frame");
  Check(source.find("backend_manager.ResetTemporal(") == std::string::npos,
        "freshly prepared temporal programs must not be reset unconditionally");
  Check(source.find("execution.output.planes[0].write_data != nullptr") !=
            std::string::npos &&
            source.find("execution.output.planes[0].read_data != nullptr") ==
                std::string::npos,
        "completed host output must consume the borrowed writable binding");
  Check(source.find("manager_output_committed && committed->valid") !=
            std::string::npos &&
            source.find(
                "committed->capture_sequence == frame.sequence") !=
                std::string::npos,
        "live attribution must reject a prior committed capture");
  Check(source.find("FrameDropBehavior::reset_temporal_state") !=
            std::string::npos &&
            source.find("requires_contiguous_capture_sequence = temporal") !=
                std::string::npos,
        "canonical request must preserve temporal ordering and drop behavior");
}

void TestCpuPrimitiveLaneAndBoundedShutdown() {
  const auto source = Read("src/core/video/camera_pipeline.cpp");
  Check(source.find("GetCpuPrimitiveTable()") != std::string::npos &&
            source.find(
                "ProductionVideoBackendSetupKind::cpu_primitives_only") !=
                std::string::npos,
        "host primitives must remain a separate CPU HAL lane");
  Check(source.find("PrepareRawYuyvPassthroughPlan(") != std::string::npos &&
            source.find("CopyRawYuyvFrame(") != std::string::npos,
        "zero-effect matching YUYV must retain CPU-only raw passthrough");
  Check(source.find("compute_backend == "
                    "ComputeBackendPreference::vulkan") !=
            std::string::npos &&
            source.find("request.resident.effects.mirror = false") !=
                std::string::npos,
        "strict Vulkan must retain resident mirror while other modes lower it");
  Check(source.find("PrimitiveOperationId::mirror") != std::string::npos &&
            source.find("cpu.mirror_rgb24_in_place(") !=
                std::string::npos &&
            source.find("mirror_fail_open_frames") != std::string::npos,
        "non-Vulkan mirror must be an explicit measured CPU primitive tail");
  const auto mirror = source.find("cpu.mirror_rgb24_in_place(");
  const auto write = source.find("writer_.WriteFrameDetailed", mirror);
  const auto commit =
      source.find("backend_manager.CommitOutput(completion, written", write);
  Check(mirror != std::string::npos && write != std::string::npos &&
            commit != std::string::npos && mirror < write && write < commit,
        "CPU mirror finalizer must run before writer-gated manager commit");
  Check(source.find("LiveEffectBackendAttribution::"
                    "LiveEffectBackendAttribution()") !=
            std::string::npos &&
            source.find("active_backends_.reserve(384)") !=
                std::string::npos,
        "bounded attribution serialization capacity must be reserved at setup");
  Check(source.find("backend_manager.Invalidate(") != std::string::npos &&
            source.find("backend_manager.Shutdown()") != std::string::npos &&
            source.find("attempt < 8 && shutdown.pending") !=
                std::string::npos,
        "shutdown must consume typed evidence with an explicit retry bound");
}

} // namespace

int main() {
  TestPublicHeaderIsBackendNeutral();
  TestPipelineOwnsNoBackendOrCompatibilityPlanner();
  TestFrameLoopUsesOnlyPreparedManagerLifecycle();
  TestCpuPrimitiveLaneAndBoundedShutdown();
  if (failures != 0) {
    std::cerr << failures << " final pipeline cutover checks failed\n";
    return 1;
  }
  std::cout << "final pipeline cutover checks passed\n";
  return 0;
}
