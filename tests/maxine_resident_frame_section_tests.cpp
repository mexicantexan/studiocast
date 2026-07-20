#include "core/maxine/resident_frame_section.h"
#include "core/maxine/testing/fake_resident_frame_executor.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

std::atomic<uint64_t> g_allocations{0};
thread_local bool g_count_allocations = false;

void *Allocate(std::size_t size) {
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    if (g_count_allocations)
      g_allocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
  }
  throw std::bad_alloc();
}

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << __func__ << ':' << __LINE__ << ": CHECK failed: "           \
                << #condition << '\n';                                          \
      return false;                                                             \
    }                                                                           \
  } while (false)

using studiocast::maxine::CpuTailKind;
using studiocast::maxine::HostRgbFrameView;
using studiocast::maxine::ResidentBoundaryResult;
using studiocast::maxine::ResidentExecutionStatus;
using studiocast::maxine::ResidentFrameKey;
using studiocast::maxine::ResidentFramePlan;
using studiocast::maxine::ResidentFrameSection;
using studiocast::maxine::ResidentInvalidationReason;
using studiocast::maxine::ResidentStageKind;
using studiocast::maxine::testing::FakeResidentFrameExecutor;

constexpr ResidentFrameKey Key(uint64_t generation = 1, uint32_t width = 2,
                               uint32_t height = 2, uintptr_t runtime = 11,
                               uintptr_t stream = 12) {
  return ResidentFrameKey{generation, width, height, runtime, stream};
}

struct HostFixture {
  std::array<uint8_t, 64> storage{};
  HostRgbFrameView View(uint32_t width = 2, uint32_t height = 2) const {
    return HostRgbFrameView{storage.data(), width, height,
                            static_cast<std::size_t>(width) * 3u};
  }
};

ResidentFramePlan CombinedPlan() {
  ResidentFramePlan plan{};
  const bool first = plan.AddCompatible(ResidentStageKind::background_blur,
                                        false, true, 77);
  const bool second =
      plan.AddCompatible(ResidentStageKind::relighting, false, true, 77);
  if (!first || !second)
    std::abort();
  return plan;
}

bool CombinedFrameUsesOneTransferPairAndOneMatte() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  ResidentFrameSection section{};
  const auto result = section.Execute(CombinedPlan(), Key(), 99, host.View(),
                                      executor);
  const auto &counters = section.counters();
  const auto &calls = executor.calls();

  CHECK(result.status == ResidentExecutionStatus::cpu_visible_output);
  CHECK(result.resident_stages_consumed == 2);
  CHECK(result.next_stage_index == 2);
  CHECK(counters.upload_attempts == 1 && counters.upload_successes == 1);
  CHECK(counters.final_download_attempts == 1 &&
        counters.final_download_successes == 1);
  CHECK(counters.cpu_continuation_download_attempts == 0);
  CHECK(counters.matte_inference_attempts == 1 &&
        counters.matte_inference_successes == 1);
  CHECK(counters.shared_matte_reuses == 1);
  CHECK(counters.sync_attempts == 1 && counters.sync_successes == 1);
  CHECK(calls.upload == 1 && calls.download == 1 && calls.synchronize == 1);
  CHECK(calls.matte == 1 && calls.stage_count == 2);
  CHECK(calls.matte_source_identity == calls.uploaded_image_identity);
  CHECK(calls.stage_input_identities[0] == calls.uploaded_image_identity);
  CHECK(calls.stage_input_identities[1] == calls.stage_output_identities[0]);
  CHECK(calls.stage_input_identities[1] != calls.matte_source_identity);
  CHECK(result.matte.source_image_identity == calls.matte_source_identity);
  return true;
}

bool EmptyPlanIsGpuFreeEvenWithInvalidInputs() {
  FakeResidentFrameExecutor executor{};
  ResidentFrameSection section{};
  ResidentFramePlan plan{};
  const auto result = section.Execute(plan, {}, 0, {}, executor);
  CHECK(result.status == ResidentExecutionStatus::no_effects_passthrough);
  CHECK(result.boundary_result == ResidentBoundaryResult::success);
  CHECK(section.counters().prepare_attempts == 0);
  CHECK(section.counters().upload_attempts == 0);
  CHECK(section.counters().download_attempts == 0);
  CHECK(section.counters().sync_attempts == 0);
  CHECK(executor.calls().prepare == 0 && executor.calls().upload == 0);
  return true;
}

bool OptionalFailurePreservesPriorResidentCurrent() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  executor.FailStage(ResidentStageKind::denoise,
                     ResidentBoundaryResult::runtime_failure);
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, true));
  CHECK(plan.AddCompatible(ResidentStageKind::transfer, false));
  plan.require_cpu_output = false;
  ResidentFrameSection section{};
  const auto result = section.Execute(plan, Key(), 1, host.View(), executor);
  CHECK(result.status == ResidentExecutionStatus::resident_output);
  CHECK(result.optional_failure_observed);
  CHECK(section.counters().optional_fail_open == 1);
  CHECK(executor.calls().stage_count == 2);
  CHECK(executor.calls().stage_input_identities[0] ==
        executor.calls().uploaded_image_identity);
  CHECK(executor.calls().stage_input_identities[1] ==
        executor.calls().uploaded_image_identity);
  return true;
}

bool CpuTailIsExplicitAndTerminal() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));
  CHECK(plan.SetCpuTail(CpuTailKind::key_light));
  CHECK(!plan.AddCompatible(ResidentStageKind::transfer, false));
  ResidentFrameSection section{};
  const auto result = section.Execute(plan, Key(), 1, host.View(), executor);
  CHECK(result.status == ResidentExecutionStatus::cpu_tail_required);
  CHECK(result.cpu_tail == CpuTailKind::key_light);
  CHECK(result.resident_stages_consumed == 1);
  CHECK(result.next_stage_index == 1);
  CHECK(section.counters().cpu_continuation_download_attempts == 1);
  CHECK(section.counters().cpu_continuation_download_successes == 1);
  CHECK(section.counters().final_download_attempts == 0);
  CHECK(section.counters().cpu_continuation_sync_successes == 1);
  CHECK(section.counters().cpu_tail_boundaries[0] == 1);
  return true;
}

bool KeysFailClosedAndChangesPrepareExactlyOnce() {
  HostFixture host{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));

  FakeResidentFrameExecutor invalid_executor{};
  ResidentFrameSection invalid_section{};
  CHECK(invalid_section.Execute(plan, Key(0), 1, host.View(), invalid_executor)
            .status == ResidentExecutionStatus::failed);
  CHECK(invalid_section.Execute(plan, Key(1, 2, 2, 11, 0), 2, host.View(),
                                invalid_executor)
            .status == ResidentExecutionStatus::failed);
  CHECK(invalid_executor.calls().prepare == 0);

  FakeResidentFrameExecutor executor{};
  ResidentFrameSection section{};
  auto key = Key();
  CHECK(section.Execute(plan, key, 1, host.View(), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  CHECK(section.Execute(plan, key, 2, host.View(), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  CHECK(executor.calls().prepare == 1);

  key.configuration_generation = 2;
  CHECK(section.Execute(plan, key, 3, host.View(), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  key.width = 3;
  HostFixture wider_host{};
  CHECK(section.Execute(plan, key, 4, wider_host.View(3, 2), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  key.runtime_identity = 21;
  CHECK(section.Execute(plan, key, 5, wider_host.View(3, 2), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  key.stream_identity = 22;
  CHECK(section.Execute(plan, key, 6, wider_host.View(3, 2), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  const auto &counters = section.counters();
  CHECK(counters.prepare_successes == 5);
  CHECK(counters.invalidations == 4);
  CHECK(counters.generation_invalidations == 1);
  CHECK(counters.geometry_invalidations == 1);
  CHECK(counters.runtime_invalidations == 1);
  CHECK(counters.stream_invalidations == 1);
  return true;
}

bool ExplicitInvalidationPreservesTelemetryAndRepreparesOnce() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));
  ResidentFrameSection section{};
  CHECK(section.Execute(plan, Key(), 1, host.View(), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  const uint64_t uploads_before = section.counters().upload_successes;
  section.Invalidate(ResidentInvalidationReason::runtime_failure);
  CHECK(!section.prepared());
  CHECK(section.counters().upload_successes == uploads_before);
  CHECK(section.counters().explicit_invalidation_requests == 1);
  CHECK(section.counters().explicit_invalidations == 1);
  CHECK(section.counters().explicit_invalidations_by_reason[3] == 1);
  CHECK(section.Execute(plan, Key(), 2, host.View(), executor).status ==
        ResidentExecutionStatus::cpu_visible_output);
  CHECK(executor.calls().prepare == 2);
  CHECK(section.counters().prepare_successes == 2);
  CHECK(section.counters().invalidations == 1);
  return true;
}

bool AttemptAndSuccessCountersDistinguishFailures() {
  HostFixture host{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));

  FakeResidentFrameExecutor upload_executor{};
  upload_executor.FailUpload(ResidentBoundaryResult::runtime_failure);
  ResidentFrameSection upload_section{};
  CHECK(upload_section.Execute(plan, Key(), 1, host.View(), upload_executor)
            .status == ResidentExecutionStatus::failed);
  CHECK(upload_section.counters().upload_attempts == 1);
  CHECK(upload_section.counters().upload_successes == 0);

  FakeResidentFrameExecutor download_executor{};
  download_executor.FailDownload(ResidentBoundaryResult::runtime_failure);
  ResidentFrameSection download_section{};
  CHECK(download_section.Execute(plan, Key(), 1, host.View(), download_executor)
            .status == ResidentExecutionStatus::failed);
  CHECK(download_section.counters().download_attempts == 1);
  CHECK(download_section.counters().download_successes == 0);
  CHECK(download_section.counters().sync_attempts == 0);

  FakeResidentFrameExecutor sync_executor{};
  sync_executor.FailSynchronize(ResidentBoundaryResult::runtime_failure);
  ResidentFrameSection sync_section{};
  CHECK(sync_section.Execute(plan, Key(), 1, host.View(), sync_executor).status ==
        ResidentExecutionStatus::failed);
  CHECK(sync_section.counters().download_successes == 1);
  CHECK(sync_section.counters().sync_attempts == 1);
  CHECK(sync_section.counters().sync_successes == 0);
  return true;
}

bool MatteValidationMissFailsClosedWithoutSecondInference() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  executor.CorruptNextMatteOutput();
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::background_blur, true, true, 7));
  CHECK(plan.AddCompatible(ResidentStageKind::relighting, true, true, 7));
  ResidentFrameSection section{};
  const auto result = section.Execute(plan, Key(), 1, host.View(), executor);
  CHECK(result.status == ResidentExecutionStatus::cpu_visible_output);
  CHECK(result.optional_failure_observed);
  CHECK(executor.calls().matte == 1);
  CHECK(executor.calls().stage_count == 0);
  CHECK(section.counters().incompatible_outputs == 1);
  CHECK(section.counters().incompatible_matte_requests == 1);
  CHECK(section.counters().optional_fail_open == 2);
  return true;
}

bool RequiredFailureAndIncompatibleOutputFailClosed() {
  HostFixture host{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));

  FakeResidentFrameExecutor failed_executor{};
  failed_executor.FailStage(ResidentStageKind::denoise,
                            ResidentBoundaryResult::runtime_failure);
  ResidentFrameSection failed_section{};
  const auto failed =
      failed_section.Execute(plan, Key(), 1, host.View(), failed_executor);
  CHECK(failed.status == ResidentExecutionStatus::failed);
  CHECK(failed.resident_stages_consumed == 0);
  CHECK(failed.next_stage_index == 0);
  CHECK(failed_section.counters().required_failures == 1);

  FakeResidentFrameExecutor incompatible_executor{};
  incompatible_executor.CorruptNextStageOutput();
  ResidentFrameSection incompatible_section{};
  const auto incompatible = incompatible_section.Execute(
      plan, Key(), 1, host.View(), incompatible_executor);
  CHECK(incompatible.status == ResidentExecutionStatus::failed);
  CHECK(incompatible.boundary_result ==
        ResidentBoundaryResult::incompatible_output);
  CHECK(incompatible_section.counters().incompatible_outputs == 1);
  CHECK(incompatible_section.counters().required_failures == 1);
  return true;
}

bool MatteFingerprintMismatchDoesNotRunAgain() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::background_blur, false, true, 7));
  CHECK(plan.AddCompatible(ResidentStageKind::relighting, true, true, 8));
  ResidentFrameSection section{};
  const auto result = section.Execute(plan, Key(), 1, host.View(), executor);
  CHECK(result.status == ResidentExecutionStatus::cpu_visible_output);
  CHECK(result.optional_failure_observed);
  CHECK(executor.calls().matte == 1);
  CHECK(executor.calls().stage_count == 1);
  CHECK(section.counters().matte_inference_attempts == 1);
  CHECK(section.counters().incompatible_matte_requests == 1);
  CHECK(section.counters().optional_fail_open == 1);
  return true;
}

bool PlanCapacityAndEnumsFailClosed() {
  ResidentFramePlan full{};
  for (std::size_t i = 0; i < studiocast::maxine::kMaxResidentFrameStages;
       ++i)
    CHECK(full.AddCompatible(ResidentStageKind::denoise, true));
  CHECK(!full.AddCompatible(ResidentStageKind::transfer, true));
  CHECK(full.Valid());
  CHECK(!full.AddCompatible(ResidentStageKind::count, true));

  ResidentFramePlan invalid_stage{};
  invalid_stage.stage_count = 1;
  invalid_stage.stages[0].kind = ResidentStageKind::count;
  CHECK(!invalid_stage.Valid());

  ResidentFramePlan invalid_tail{};
  CHECK(!invalid_tail.SetCpuTail(CpuTailKind::count));
  CHECK(!invalid_tail.SetCpuTail(CpuTailKind::key_light));
  return true;
}

bool ResidentOutputHasNoReadbackOrSynchronization() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));
  plan.require_cpu_output = false;
  ResidentFrameSection section{};
  const auto result = section.Execute(plan, Key(), 1, host.View(), executor);
  CHECK(result.status == ResidentExecutionStatus::resident_output);
  CHECK(section.counters().upload_successes == 1);
  CHECK(section.counters().download_attempts == 0);
  CHECK(section.counters().sync_attempts == 0);
  CHECK(executor.calls().download == 0 && executor.calls().synchronize == 0);
  return true;
}

bool SteadyExecutionHasNoApplicationHeapAllocations() {
  HostFixture host{};
  FakeResidentFrameExecutor executor{};
  ResidentFrameSection section{};
  const auto plan = CombinedPlan();
  for (uint64_t i = 0; i < 16; ++i)
    CHECK(section.Execute(plan, Key(), i + 1, host.View(), executor).status ==
          ResidentExecutionStatus::cpu_visible_output);

  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool all_succeeded = true;
  for (uint64_t i = 0; i < 128; ++i) {
    if (section.Execute(plan, Key(), i + 100, host.View(), executor).status !=
        ResidentExecutionStatus::cpu_visible_output) {
      all_succeeded = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(all_succeeded);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  CHECK(section.counters().prepare_successes == 1);
  return true;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

int main() {
  const std::array tests{
      CombinedFrameUsesOneTransferPairAndOneMatte,
      EmptyPlanIsGpuFreeEvenWithInvalidInputs,
      OptionalFailurePreservesPriorResidentCurrent,
      CpuTailIsExplicitAndTerminal,
      KeysFailClosedAndChangesPrepareExactlyOnce,
      ExplicitInvalidationPreservesTelemetryAndRepreparesOnce,
      AttemptAndSuccessCountersDistinguishFailures,
      MatteValidationMissFailsClosedWithoutSecondInference,
      RequiredFailureAndIncompatibleOutputFailClosed,
      MatteFingerprintMismatchDoesNotRunAgain,
      PlanCapacityAndEnumsFailClosed,
      ResidentOutputHasNoReadbackOrSynchronization,
      SteadyExecutionHasNoApplicationHeapAllocations,
  };
  for (const auto test : tests) {
    if (!test())
      return EXIT_FAILURE;
  }
  std::cout << "maxine resident frame section tests passed\n";
  return EXIT_SUCCESS;
}
