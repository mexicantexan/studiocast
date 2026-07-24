#include "core/video/camera_pipeline.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

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

std::uint64_t Align4096(std::uint64_t value) {
  return ((value + 4095u) / 4096u) * 4096u;
}

void TestOddModelGeometryBudget() {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  hal::ResourceBudget budget{};
  CHECK(video::BuildPreparedOpenCudaPipelineResourceBudget(
      641, 479, 257, 263,
      hal::SemanticEffectId::virtual_background_remove, 8, &budget));

  const std::uint64_t frame_rgb = Align4096(641u * 3u) * 479u;
  const std::uint64_t frame_alpha = Align4096(641u * 4u) * 479u;
  const std::uint64_t model_alpha = Align4096(257u * 4u) * 263u;
  const std::uint64_t model_tensor = Align4096(257u * 12u) * 263u;
  const std::uint64_t expected_persistent =
      frame_rgb * 2u + frame_alpha + model_alpha + model_tensor;
  CHECK(budget.persistent_bytes == expected_persistent);
  CHECK(budget.scratch_bytes == 1);
  CHECK(budget.upload_staging_bytes == 641u * 479u * 3u);
  CHECK(budget.readback_staging_bytes == 641u * 479u * 3u);
  CHECK(budget.resource_count == 17);
  CHECK(budget.in_flight_slots == 1);

  CHECK(video::BuildPreparedOpenCudaPipelineResourceBudget(
      641, 479, 257, 263,
      hal::SemanticEffectId::virtual_background_blur, 64, &budget));
  CHECK(budget.scratch_bytes ==
        frame_alpha * 2u + frame_rgb * 2u);
  CHECK(budget.resource_count == 21);

  CHECK(!video::BuildPreparedOpenCudaPipelineResourceBudget(
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max(), 257, 263,
      hal::SemanticEffectId::virtual_background_blur, 64, &budget));
}

void TestWriterCommitAttributionBoundary() {
  namespace hal = studiocast::hal;
  namespace video = studiocast::video;
  CHECK(video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::succeeded, true, hal::StatusId::ready));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::succeeded, false, hal::StatusId::ready));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::succeeded, true, hal::StatusId::failed));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::optional_failed_open, true,
      hal::StatusId::ready));
  CHECK(!video::PreparedOpenCudaPipelineMayCommitAttribution(
      hal::StageExecutionStatus::required_failed, true,
      hal::StatusId::ready));

  CHECK(!video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::failed, hal::ReasonId::incomplete_execution));
  CHECK(!video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::degraded, hal::ReasonId::optional_stage_failed_open));
  CHECK(video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::quarantined, hal::ReasonId::device_lost));
  CHECK(video::PreparedOpenCudaPipelineShouldSuspend(
      hal::StatusId::stopped, hal::ReasonId::shutdown_requested));
}

std::string ReadPipelineSource() {
  const auto root = std::filesystem::path(STUDIOCAST_SOURCE_DIR);
  std::ifstream input(root / "src/core/video/camera_pipeline.cpp");
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

void TestMigrationStructuralGuards() {
  const std::string source = ReadPipelineSource();
  CHECK(!source.empty());
  if (source.empty())
    return;

  const auto setup = source.find(
      "try_prepare_open_cuda_virtual_background");
  const auto loop = source.find("while (!stop_.load())");
  const auto execute = source.find(
      "prepared_open_cuda_hardware_owner.ExecuteHostFrame");
  const auto write = source.find("writer_.WriteFrameDetailed");
  const auto commit_gate = source.find(
      "PreparedOpenCudaPipelineMayCommitAttribution", write);
  CHECK(setup != std::string::npos && loop != std::string::npos &&
        setup < loop);
  CHECK(execute != std::string::npos && execute > loop);
  CHECK(write != std::string::npos && commit_gate != std::string::npos &&
        write < commit_gate);

  CHECK(source.find(
            "fx.engine == effects::EffectsEnginePreference::open_cuda") !=
        std::string::npos);
  CHECK(source.find("request.compute = PreparedOpenCudaComputeIntent::cuda") !=
        std::string::npos);
  CHECK(source.find(
            "request.semantic_stage_count = plan.ordered_effect_ids.size()") !=
        std::string::npos);
  CHECK(source.find("request.tensorrt_enabled = "
                    "OpenCudaTensorRtEnabledFromEnv()") !=
        std::string::npos);
  CHECK(source.find("outA.format == PixelFormat::rgb24") !=
        std::string::npos);
  CHECK(source.find("request.scaling_required =") != std::string::npos);
  CHECK(source.find("request.output_format_conversion_required =") !=
        std::string::npos);
  CHECK(source.find("ValidateModelArtifactIdentity") != std::string::npos);
  CHECK(source.find("prepared_open_cuda_hardware_owner.ObserveDeviceLoss") !=
        std::string::npos);
  CHECK(source.find("prepared_open_cuda_hardware_owner.Shutdown") !=
        std::string::npos);
  CHECK(source.find("Never retry through the legacy context") !=
        std::string::npos);
  CHECK(source.find("breaker.AllowsAttempt(capture_sequence)") !=
        std::string::npos);

  // Provider probing, model scanning, artifact hashing, Configure and JSON
  // must remain before the capture loop. ExecuteHostFrame is the sole facade
  // call in the live section.
  const std::string live = source.substr(loop);
  CHECK(live.find("ModelPackRegistry::ScanDefault") == std::string::npos);
  CHECK(live.find("ValidateModelArtifactIdentity") == std::string::npos);
  CHECK(live.find("prepared_open_cuda_hardware_owner.Configure") ==
        std::string::npos);
  CHECK(live.find("Json") == std::string::npos);
  CHECK(live.find("JSON") == std::string::npos);
}

} // namespace

int main() {
  TestOddModelGeometryBudget();
  TestWriterCommitAttributionBoundary();
  TestMigrationStructuralGuards();
  if (failures != 0) {
    std::cerr << failures
              << " prepared Open CUDA pipeline migration checks failed\n";
    return 1;
  }
  std::cout << "prepared Open CUDA pipeline migration checks passed\n";
  return 0;
}
