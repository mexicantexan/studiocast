#include "core/open_video/prepared_open_video_compatibility_executor.h"
#include "core/video/cpu_primitive_hal.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

std::atomic<bool> g_fail_allocations{false};

void *Allocate(std::size_t size) {
  if (g_fail_allocations.load(std::memory_order_relaxed))
    throw std::bad_alloc();
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

namespace hal = studiocast::hal;
namespace open_video = studiocast::open_video;
namespace video = studiocast::video;

constexpr std::size_t kFrameBytes = 24;

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ImageDescriptor Image(hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = 4;
  image.height = 2;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.alpha_meaning = hal::AlphaMeaning::none;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::coherent;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = 12;
  image.planes[0].size_bytes = kFrameBytes;
  image.planes[0].width = 4;
  image.planes[0].height = 2;
  image.planes[0].alignment_bytes = 1;
  return image;
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::open_video_onnxruntime;
  domain.api_family = hal::ApiFamily::onnxruntime;
  domain.physical_device.vendor_id = 0x535443;
  domain.physical_device.stable_id[0] = 0x4f;
  domain.logical_context_generation = 1;
  domain.allocator_family = hal::AllocatorFamily::onnxruntime;
  domain.memory_handle_family = hal::MemoryHandleFamily::ort_value;
  domain.queue_or_stream_identity = 17;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::ort_provider_synchronous;
  domain.adapter_instance_id = 19;
  return domain;
}

struct RunnerState {
  std::uint64_t calls = 0;
  bool fail_required = false;
  bool succeed_optional = false;
  int width = 0;
  int height = 0;
  std::size_t stride = 0;
};

bool RunStage(void *opaque, const video::CanonicalVideoStage &stage,
              std::uint64_t, std::uint8_t *rgb, int width, int height,
              std::size_t stride, std::string *error) noexcept {
  auto *state = static_cast<RunnerState *>(opaque);
  ++state->calls;
  state->width = width;
  state->height = height;
  state->stride = stride;
  if (stage.intent.semantic == hal::SemanticEffectId::video_noise_removal) {
    rgb[0] = 0x22;
    return !state->fail_required;
  }
  if (state->succeed_optional) {
    rgb[1] = 0x33;
    return true;
  }
  rgb[0] = 0xff;
  if (error != nullptr)
    *error = "prepared optional stage failed after partial mutation";
  return false;
}

open_video::PreparedOpenVideoCompatibilityConfig Config(RunnerState *state) {
  open_video::PreparedOpenVideoCompatibilityConfig config;
  config.plan.plan_id = 77;
  config.plan.budget.persistent_bytes = 1u << 20;
  config.plan.budget.scratch_bytes = 1u << 20;
  config.plan.budget.resource_count = 8;
  config.plan.budget.in_flight_slots = 1;
  config.plan.host_input = Image(hal::Mutability::read_only);
  config.plan.host_output = Image(hal::Mutability::write_only);
  config.plan.intent.backend.backend =
      hal::BackendId::open_video_onnxruntime;
  config.plan.intent.stages[0].intent.stage_id = 1;
  config.plan.intent.stages[0].intent.semantic =
      hal::SemanticEffectId::video_noise_removal;
  config.plan.intent.stages[0].intent.required = true;
  config.plan.intent.stages[1].intent.stage_id = 2;
  config.plan.intent.stages[1].intent.semantic =
      hal::SemanticEffectId::eye_contact;
  config.plan.intent.stages[1].intent.required = false;
  config.plan.intent.stages[1].intent.failure_action =
      hal::FailureAction::fail_open_preserve_input;
  config.plan.intent.stage_count = 2;
  config.providers[0] =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  config.providers[1] =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  config.backends[0] = hal::BackendId::open_video_onnxruntime;
  config.backends[1] = hal::BackendId::open_video_onnxruntime;
  config.selected_backend = hal::BackendId::open_video_onnxruntime;
  config.selected_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  config.execution_domain = Domain();
  config.prepared_stage_runner = &RunStage;
  config.prepared_stage_runner_context = state;
  return config;
}

hal::HostFrameBinding
Binding(const std::array<std::byte, kFrameBytes> &input,
        std::array<std::byte, kFrameBytes> *output,
        std::uint64_t capture_sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = Image(hal::Mutability::read_only);
  binding.output_descriptor = Image(hal::Mutability::write_only);
  binding.input_planes[0].read_data = input.data();
  binding.input_planes[0].span_bytes = input.size();
  binding.input_planes[0].row_stride_bytes = 12;
  binding.output_planes[0].write_data = output->data();
  binding.output_planes[0].span_bytes = output->size();
  binding.output_planes[0].row_stride_bytes = 12;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = capture_sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  return binding;
}

bool TestFailOpenPreservesAndExecuteDoesNotAllocate() {
  RunnerState state;
  open_video::PreparedOpenVideoCompatibilitySetupResult setup;
  auto executor =
      open_video::PreparedOpenVideoCompatibilityExecutor::Create(Config(&state),
                                                                 &setup);
  if (!Require(executor != nullptr, "prepared executor setup must succeed"))
    return false;

  std::array<std::byte, kFrameBytes> input{};
  std::array<std::byte, kFrameBytes> writer_output{};
  input[0] = std::byte{0x11};
  g_fail_allocations.store(true, std::memory_order_relaxed);
  const auto frame = executor->Execute(Binding(input, &writer_output, 9));
  g_fail_allocations.store(false, std::memory_order_relaxed);

  bool ok = Require(frame.status == hal::StatusId::degraded &&
                        frame.reason ==
                            hal::ReasonId::optional_stage_failed_open &&
                        frame.output_ready && frame.stage_count == 2 &&
                        frame.stages[1].status ==
                            hal::StageExecutionStatus::optional_failed_open &&
                        frame.stages[1].valid_output_preserved,
                    "optional failure must be typed and preserve valid output");
  ok &= Require(frame.output.planes[0].read_data[0] == std::byte{0x22},
                "partial optional-stage mutation must not escape candidate");
  ok &= Require(state.calls == 2,
                "each prepared stage must execute exactly once");

  auto forged = frame.completion;
  ++forged.program_id;
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(executor->CommitOutput(forged, true, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::output_not_committed,
                "commit must reject a foreign program identity");
  forged = frame.completion;
  ++forged.adapter_instance_id;
  ok &= Require(executor->CommitOutput(forged, true, &reason) ==
                        hal::StatusId::blocked,
                "commit must reject a foreign adapter identity");
  forged = frame.completion;
  forged.complete = false;
  ok &= Require(executor->CommitOutput(forged, true, &reason) ==
                        hal::StatusId::blocked,
                "commit must reject a non-complete token");
  ok &= Require(executor->CommitOutput(frame.completion, true, &reason) ==
                        hal::StatusId::ready &&
                    executor->Snapshot().latest_committed_frame.valid,
                "exact pending completion must commit");
  return ok;
}

bool TestFailedPublicationClearsCurrentAttributionOnly() {
  RunnerState state;
  open_video::PreparedOpenVideoCompatibilitySetupResult setup;
  auto executor = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
      Config(&state), &setup);
  if (!Require(executor != nullptr, "publication fixture must prepare"))
    return false;
  std::array<std::byte, kFrameBytes> input{}, output{};
  hal::ReasonId reason = hal::ReasonId::none;
  const auto first = executor->Execute(Binding(input, &output, 1));
  if (!Require(first.output_ready &&
                   executor->CommitOutput(first.completion, true, &reason) ==
                       hal::StatusId::ready,
               "first compatibility publication must succeed"))
    return false;
  const auto selected = executor->Snapshot();
  const auto second = executor->Execute(Binding(input, &output, 2));
  const auto calls = state.calls;
  const auto duplicate = executor->Execute(Binding(input, &output, 99));
  bool ok = Require(
      duplicate.status == hal::StatusId::blocked && state.calls == calls &&
          executor->Snapshot().latest_committed_frame.valid,
      "duplicate Execute must preserve pending/current publication");
  std::array<hal::CompletionToken, 5> forged{};
  forged.fill(second.completion);
  ++forged[0].token_id;
  ++forged[1].program_id;
  ++forged[2].adapter_instance_id;
  ++forged[3].capture_sequence;
  forged[4].complete = false;
  for (const auto &token : forged) {
    ok &= Require(
        second.output_ready &&
            executor->CommitOutput(token, false, &reason) ==
                hal::StatusId::blocked &&
            reason == hal::ReasonId::output_not_committed &&
            executor->Snapshot().latest_committed_frame.valid &&
            executor->Snapshot().latest_committed_frame.capture_sequence == 1 &&
            executor->Snapshot().telemetry.active_provider ==
                selected.telemetry.active_provider,
        "forged completion must preserve pending and current publication");
  }
  ok &= Require(
      executor->CommitOutput(second.completion, false, &reason) ==
              hal::StatusId::failed &&
          reason == hal::ReasonId::output_write_failed,
      "exact failed writer must consume pending compatibility completion");
  const auto failed = executor->Snapshot();
  ok &=
      Require(!failed.latest_committed_frame.valid &&
                  failed.telemetry.active_backend == hal::BackendId::invalid &&
                  failed.telemetry.active_provider ==
                      hal::SemanticEffectProviderId::invalid &&
                  failed.telemetry.committed_output_frames == 1 &&
                  failed.telemetry.active_backend_frames == 1 &&
                  failed.prepared && failed.plan_id == selected.plan_id &&
                  failed.readiness.available == selected.readiness.available &&
                  failed.readiness.status == selected.readiness.status &&
                  failed.readiness.execution_domain ==
                      selected.readiness.execution_domain &&
                  failed.telemetry.selected_backend ==
                      selected.telemetry.selected_backend,
              "failed compatibility publication must clear current activity "
              "but retain selected history");
  ok &= Require(executor->CommitOutput(second.completion, true, &reason) ==
                        hal::StatusId::blocked &&
                    executor->Snapshot().telemetry.committed_output_frames == 1,
                "failed compatibility completion must not later publish");
  const auto third = executor->Execute(Binding(input, &output, 3));
  ok &= Require(
      third.output_ready &&
          executor->CommitOutput(third.completion, true, &reason) ==
              hal::StatusId::ready &&
          executor->Snapshot().latest_committed_frame.valid &&
          executor->Snapshot().latest_committed_frame.capture_sequence == 3 &&
          executor->Snapshot().latest_committed_frame.output_commit_sequence ==
              2 &&
          executor->Snapshot().telemetry.committed_output_frames == 2 &&
          executor->Snapshot().telemetry.active_backend_frames == 2 &&
          executor->Snapshot().telemetry.active_provider ==
              selected.telemetry.active_provider,
      "next exact compatibility publication must recover current attribution");
  return ok;
}

bool TestExecutionFailureAndCleanupClearOnlyCurrentActivity() {
  bool ok = true;
  for (const int boundary :
       {0, 1, 2}) { // Required runner failure, invalidation, shutdown.
    RunnerState state;
    open_video::PreparedOpenVideoCompatibilitySetupResult setup;
    auto executor = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
        Config(&state), &setup);
    if (!Require(executor != nullptr, "activity fixture must prepare"))
      return false;
    std::array<std::byte, kFrameBytes> input{}, output{};
    hal::ReasonId reason = hal::ReasonId::none;
    const auto first = executor->Execute(Binding(input, &output, 1));
    if (!Require(first.output_ready &&
                     executor->CommitOutput(first.completion, true, &reason) ==
                         hal::StatusId::ready,
                 "activity fixture must first publish"))
      return false;
    const auto selected = executor->Snapshot();
    std::array<hal::HostFrameBinding, 6> malformed{};
    malformed.fill(Binding(input, &output, 2));
    malformed[0].input_plane_count = 0;
    malformed[1].input_planes[0].read_data = nullptr;
    malformed[2].capture_sequence = 0;
    ++malformed[3].input_descriptor.width;
    --malformed[4].input_planes[0].span_bytes;
    ++malformed[5].input_planes[0].row_stride_bytes;
    const auto calls = state.calls;
    for (const auto &binding : malformed) {
      const auto rejected = executor->Execute(binding);
      ok &= Require(
          rejected.status == hal::StatusId::blocked &&
              rejected.reason == hal::ReasonId::host_frame_binding_invalid &&
              state.calls == calls &&
              executor->Snapshot().latest_committed_frame.valid &&
              executor->Snapshot().latest_committed_frame.capture_sequence ==
                  1 &&
              executor->Snapshot().telemetry.active_backend ==
                  selected.telemetry.active_backend,
          "foreign binding must not erase current publication or enter stages");
    }
    if (boundary == 0) {
      state.fail_required = true;
      const auto failed = executor->Execute(Binding(input, &output, 2));
      ok &= Require(failed.status == hal::StatusId::failed &&
                        failed.reason == hal::ReasonId::required_stage_failed &&
                        !failed.output_ready && state.calls == calls + 1,
                    "required stage must expose real execution failure");
    } else if (boundary == 1) {
      executor->Invalidate(hal::ReasonId::reconfiguration_required);
    } else {
      executor->Shutdown();
    }
    const auto failed = executor->Snapshot();
    ok &= Require(
        !failed.latest_committed_frame.valid &&
            failed.telemetry.active_backend == hal::BackendId::invalid &&
            failed.telemetry.active_provider ==
                hal::SemanticEffectProviderId::invalid &&
            failed.telemetry.committed_output_frames == 1 &&
            failed.telemetry.active_backend_frames == 1 &&
            failed.telemetry.selected_backend ==
                selected.telemetry.selected_backend &&
            failed.readiness.selected_backend ==
                selected.readiness.selected_backend &&
            failed.readiness.provider == selected.readiness.provider &&
            failed.plan_id == selected.plan_id,
        "real failure/cleanup must clear current activity and retain selected "
        "history");
    if (boundary == 0) {
      ok &= Require(failed.prepared &&
                        failed.readiness.available ==
                            selected.readiness.available &&
                        failed.readiness.status == selected.readiness.status &&
                        failed.telemetry.required_stage_failures == 1,
                    "failed frame must preserve prepared retry eligibility");
      state.fail_required = false;
      const auto recovered = executor->Execute(Binding(input, &output, 3));
      ok &= Require(
          recovered.output_ready &&
              executor->CommitOutput(recovered.completion, true, &reason) ==
                  hal::StatusId::ready &&
              executor->Snapshot().latest_committed_frame.capture_sequence ==
                  3 &&
              executor->Snapshot()
                      .latest_committed_frame.output_commit_sequence == 2 &&
              executor->Snapshot().telemetry.committed_output_frames == 2,
          "next valid frame must recover after runner failure");
    } else {
      const auto expected =
          boundary == 1 ? hal::StatusId::stale : hal::StatusId::stopped;
      ok &=
          Require(!failed.prepared && !failed.readiness.available &&
                      failed.readiness.status == expected &&
                      executor->Execute(Binding(input, &output, 3)).status ==
                          expected &&
                      state.calls == calls &&
                      executor->CommitOutput(first.completion, true, &reason) ==
                          hal::StatusId::blocked,
                  "invalidated/stopped owner must stay closed without stale "
                  "publication");
      auto replacement =
          open_video::PreparedOpenVideoCompatibilityExecutor::Create(
              Config(&state), &setup);
      if (!replacement)
        return false;
      const auto recovered = replacement->Execute(Binding(input, &output, 3));
      ok &= Require(
          recovered.output_ready &&
              replacement->CommitOutput(recovered.completion, true, &reason) ==
                  hal::StatusId::ready,
          "fresh setup must restore publication after cleanup");
    }
  }
  return ok;
}

bool TestCameraWriterAcceptsProcessedOwnedOutput() {
  const auto camera_accepts =
      [](const video::VideoExecutionFrameResult &execution) {
        return (execution.status == hal::StatusId::executed ||
                execution.status == hal::StatusId::degraded) &&
               execution.output_ready && execution.output.output_ready &&
               execution.output.plane_count == 1 &&
               execution.output.planes[0].write_data != nullptr;
      };
  for (const bool optional_succeeds : {false, true}) {
    RunnerState state;
    state.succeed_optional = optional_succeeds;
    open_video::PreparedOpenVideoCompatibilitySetupResult setup;
    const auto config = Config(&state);
    auto executor = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
        config, &setup);
    if (!Require(executor != nullptr, "camera transport fixture must prepare"))
      return false;
    std::array<std::byte, kFrameBytes> input{}, writer_output{};
    input.fill(std::byte{0x11});
    g_fail_allocations.store(true, std::memory_order_relaxed);
    const auto frame = executor->Execute(Binding(input, &writer_output, 1));
    g_fail_allocations.store(false, std::memory_order_relaxed);
    if (!Require(camera_accepts(frame),
                 "camera must accept processed compatibility output through "
                 "writable host view"))
      return false;
    const auto &view = frame.output.planes[0];
    bool ok = Require(
        state.calls == 2 && view.read_data == view.write_data &&
            view.write_data != input.data() &&
            view.write_data != writer_output.data() &&
            view.write_data[0] == std::byte{0x22} &&
            input[0] == std::byte{0x11} &&
            view.span_bytes == config.plan.host_output.planes[0].size_bytes &&
            view.row_stride_bytes ==
                config.plan.host_output.planes[0].row_stride_bytes &&
            frame.output.descriptor.width == config.plan.host_output.width &&
            frame.output.descriptor.height == config.plan.host_output.height &&
            frame.output.descriptor.mutability ==
                config.plan.host_output.mutability &&
            frame.output.capture_sequence == 1,
        "camera transport must expose owned processed storage with exact "
        "shape/span/stride");
    const auto duplicate = executor->Execute(Binding(input, &writer_output, 2));
    auto forged = frame.completion;
    forged.complete = false;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(
        duplicate.status == hal::StatusId::blocked && state.calls == 2 &&
            executor->CommitOutput(forged, false, &reason) ==
                hal::StatusId::blocked &&
            view.write_data[0] == std::byte{0x22},
        "owned output must remain valid until exact completion is consumed");
    ok &= Require(
        executor->CommitOutput(frame.completion, true, &reason) ==
                hal::StatusId::ready &&
            executor->Snapshot().latest_committed_frame.valid &&
            executor->Snapshot().latest_committed_frame.capture_sequence == 1 &&
            executor->Snapshot().telemetry.committed_output_frames == 1,
        "accepted processed camera output must commit exact current activity");
    const auto second = executor->Execute(Binding(input, &writer_output, 2));
    ok &= Require(
        camera_accepts(second) &&
            second.output.planes[0].write_data[0] == std::byte{0x22} &&
            executor->CommitOutput(second.completion, true, &reason) ==
                hal::StatusId::ready &&
            executor->Snapshot().latest_committed_frame.capture_sequence == 2 &&
            executor->Snapshot().telemetry.committed_output_frames == 2,
        "next processed frame must reuse valid writer transport and advance "
        "history");
    if (!ok)
      return false;
  }
  return true;
}

bool TestTransportDomainAndCompletionOwner() {
  RunnerState first_state, second_state;
  auto first_config = Config(&first_state),
       second_config = Config(&second_state);
  ++second_config.execution_domain.adapter_instance_id;
  open_video::PreparedOpenVideoCompatibilitySetupResult setup;
  auto first = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
      first_config, &setup);
  auto second = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
      second_config, &setup);
  if (!first || !second)
    return false;
  std::array<std::byte, kFrameBytes> input{}, writer{};
  const auto first_frame = first->Execute(Binding(input, &writer, 1));
  const auto second_frame = second->Execute(Binding(input, &writer, 1));
  bool ok = Require(
      first_frame.execution_domain == first_config.execution_domain &&
          second_frame.execution_domain == second_config.execution_domain &&
          first_frame.completion.adapter_instance_id ==
              first_config.execution_domain.adapter_instance_id &&
          second_frame.completion.adapter_instance_id ==
              second_config.execution_domain.adapter_instance_id &&
          !first_frame.passthrough && !second_frame.passthrough &&
          first_frame.transfers.uploads == 0 &&
          first_frame.transfers.final_readbacks == 0 &&
          first_frame.transfers.intermediate_uploads == 0 &&
          first_frame.transfers.intermediate_readbacks == 0 &&
          first_frame.transfers.synchronizations == 0,
      "processed frame transport must retain exact prepared execution "
      "domain/owner and host transfer flags");
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(
      second->CommitOutput(first_frame.completion, false, &reason) ==
              hal::StatusId::blocked &&
          second->CommitOutput(second_frame.completion, true, &reason) ==
              hal::StatusId::ready &&
          first->CommitOutput(first_frame.completion, true, &reason) ==
              hal::StatusId::ready,
      "foreign owner completion must not consume a legitimate pending output");
  const auto next = second->Execute(Binding(input, &writer, 2));
  auto forged = next.completion;
  forged.complete = false;
  ok &= Require(
      second->CommitOutput(forged, false, &reason) == hal::StatusId::blocked &&
          second->Snapshot().latest_committed_frame.capture_sequence == 1 &&
          second->CommitOutput(next.completion, true, &reason) ==
              hal::StatusId::ready &&
          second->Snapshot().telemetry.committed_output_frames == 2,
      "forged completion bit must preserve current and exact pending owner "
      "output");
  auto generation_config = second_config;
  ++generation_config.plan.plan_id;
  ++generation_config.execution_domain.adapter_instance_id;
  ++generation_config.execution_domain.logical_context_generation;
  auto generation = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
      generation_config, &setup);
  if (!generation)
    return false;
  const auto generation_frame = generation->Execute(Binding(input, &writer, 1));
  ok &= Require(
      generation_frame.execution_domain == generation_config.execution_domain &&
          generation->CommitOutput(second_frame.completion, false, &reason) ==
              hal::StatusId::blocked &&
          generation->CommitOutput(generation_frame.completion, true,
                                   &reason) == hal::StatusId::ready &&
          generation->Snapshot().latest_committed_frame.execution_domain ==
              generation_config.execution_domain,
      "prior generation token must not consume next prepared generation "
      "publication");
  return ok;
}

template <typename Work> bool ChildSucceeds(Work work, const char *message) {
  const auto child = fork();
  if (child == 0) {
    const rlimit limit{0, 0};
    setrlimit(RLIMIT_CORE, &limit);
    _exit(work() ? 0 : 1);
  }
  if (child < 0)
    return Require(false, "fork failed for bounded transport fixture");
  int status = 0;
  if (waitpid(child, &status, 0) != child)
    return Require(false, "transport child did not complete");
  return Require(WIFEXITED(status) && WEXITSTATUS(status) == 0, message);
}

hal::ImageDescriptor SizedImage(std::uint32_t width, std::uint32_t height,
                                std::size_t stride, std::size_t span,
                                hal::Mutability mutability) {
  auto image = Image(mutability);
  image.width = image.planes[0].width = width;
  image.height = image.planes[0].height = height;
  image.planes[0].row_stride_bytes = stride;
  image.planes[0].size_bytes = span;
  return image;
}

bool TestGuardedCaptureUpscale() {
  bool ok = true;
  for (const bool padded : {false, true})
    ok &= ChildSucceeds(
        [padded] {
          const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
          auto *mapping = static_cast<std::byte *>(
              mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
          if (mapping == MAP_FAILED ||
              mprotect(mapping + page, page, PROT_NONE) != 0)
            return false;
          const auto capture_size = padded ? 21u : kFrameBytes;
          auto *capture = mapping + page - capture_size;
          for (std::size_t i = 0; i < capture_size; ++i)
            capture[i] = std::byte{0x11};
          RunnerState state;
          auto config = Config(&state);
          if (padded)
            config.plan.host_input =
                SizedImage(3, 2, 12, capture_size, hal::Mutability::read_only);
          config.plan.host_output =
              SizedImage(8, 4, 24, 96, hal::Mutability::write_only);
          open_video::PreparedOpenVideoCompatibilitySetupResult setup;
          auto executor =
              open_video::PreparedOpenVideoCompatibilityExecutor::Create(
                  config, &setup);
          if (!executor)
            return false;
          std::array<std::byte, kFrameBytes> unused{};
          auto binding = Binding(unused, &unused, 1);
          binding.input_descriptor = config.plan.host_input;
          binding.input_planes[0].read_data = capture;
          binding.input_planes[0].span_bytes = capture_size;
          binding.output_descriptor = config.plan.host_output;
          std::array<std::byte, 96> writer{};
          binding.output_planes[0] = {nullptr, writer.data(), writer.size(),
                                      24};
          const auto frame = executor->Execute(binding);
          hal::ReasonId reason = hal::ReasonId::none;
          return frame.output_ready &&
                 frame.output.planes[0].write_data != nullptr &&
                 frame.output.planes[0].span_bytes == 96 &&
                 state.width == (padded ? 3 : 4) && state.height == 2 &&
                 state.stride == 12 &&
                 executor->CommitOutput(frame.completion, true, &reason) ==
                     hal::StatusId::ready;
        },
        "upscale must not overread tight or minimally padded "
        "guard-page-bounded capture buffers");
  return ok;
}

bool TestCaptureEffectsThenPreparedWriterResize() {
  bool ok = true;
  // Upscale, downscale, odd geometry, and equal-size padded stride/span
  // transport.
  for (const bool optional_succeeds : {false, true})
    for (const auto shape : {std::array<std::size_t, 6>{4, 2, 12, 9, 5, 27},
                             {7, 5, 21, 3, 3, 9},
                             {5, 3, 19, 3, 5, 11},
                             {5, 3, 19, 5, 3, 20}}) {
      ok &= ChildSucceeds(
          [shape, optional_succeeds] {
            const auto [iw, ih, is, ow, oh, os] = shape;
            const auto input_size = is * (ih - 1) + iw * 3;
            const auto output_size = os * (oh - 1) + ow * 3;
            RunnerState state;
            state.succeed_optional = optional_succeeds;
            auto config = Config(&state);
            config.plan.host_input = SizedImage(
                static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih),
                is, input_size, hal::Mutability::read_only);
            config.plan.host_output = SizedImage(
                static_cast<std::uint32_t>(ow), static_cast<std::uint32_t>(oh),
                os, output_size, hal::Mutability::write_only);
            std::vector<std::byte> input(input_size, std::byte{0xa5}),
                writer(output_size);
            for (std::size_t y = 0; y < ih; ++y)
              for (std::size_t x = 0; x < iw * 3; ++x)
                input[y * is + x] =
                    static_cast<std::byte>((y * 31 + x * 7) % 239);
            auto reference_input = input;
            reference_input[0] = std::byte{0x22};
            if (optional_succeeds)
              reference_input[1] = std::byte{0x33};
            const auto &cpu = video::GetCpuPrimitiveTable();
            std::vector<std::byte> scratch(
                cpu.resize_rgb24_bilinear_plan_scratch_bytes(
                    static_cast<int>(iw), static_cast<int>(ih),
                    static_cast<int>(ow), static_cast<int>(oh)));
            std::vector<std::byte> reference(os * oh);
            video::CpuRgb24BilinearResizePlan plan;
            if (cpu.prepare_resize_rgb24_bilinear(
                    static_cast<int>(iw), static_cast<int>(ih),
                    static_cast<int>(ow), static_cast<int>(oh), scratch.data(),
                    scratch.size(),
                    &plan) != video::CpuPrimitiveResult::success ||
                cpu.apply_resize_rgb24_bilinear(
                    &plan,
                    reinterpret_cast<const std::uint8_t *>(
                        reference_input.data()),
                    is, reinterpret_cast<std::uint8_t *>(reference.data()),
                    os) != video::CpuPrimitiveResult::success)
              return false;
            open_video::PreparedOpenVideoCompatibilitySetupResult setup;
            auto executor =
                open_video::PreparedOpenVideoCompatibilityExecutor::Create(
                    config, &setup);
            if (!executor)
              return false;
            hal::HostFrameBinding binding;
            binding.input_descriptor = config.plan.host_input;
            binding.output_descriptor = config.plan.host_output;
            binding.input_plane_count = binding.output_plane_count = 1;
            binding.input_planes[0] = {input.data(), nullptr, input.size(), is};
            binding.output_planes[0] = {nullptr, writer.data(), writer.size(),
                                        os};
            hal::ReasonId reason = hal::ReasonId::none;
            for (std::uint64_t sequence = 1; sequence <= 3; ++sequence) {
              binding.capture_sequence = sequence;
              g_fail_allocations.store(true, std::memory_order_relaxed);
              const auto frame = executor->Execute(binding);
              g_fail_allocations.store(false, std::memory_order_relaxed);
              if (!frame.output_ready || !frame.output.output_ready ||
                  frame.output.plane_count != 1 ||
                  !frame.output.planes[0].write_data ||
                  frame.output.planes[0].read_data !=
                      frame.output.planes[0].write_data ||
                  frame.output.descriptor.width != ow ||
                  frame.output.descriptor.height != oh ||
                  frame.output.planes[0].span_bytes != output_size ||
                  frame.output.planes[0].row_stride_bytes != os ||
                  state.width != static_cast<int>(iw) ||
                  state.height != static_cast<int>(ih) || state.stride != is)
                return false;
              for (std::size_t i = 0; i < output_size; ++i)
                if (frame.output.planes[0].read_data[i] != reference[i])
                  return false;
              if (executor->CommitOutput(frame.completion, true, &reason) !=
                      hal::StatusId::ready ||
                  executor->Snapshot()
                          .latest_committed_frame.capture_sequence !=
                      sequence ||
                  executor->Snapshot().telemetry.committed_output_frames !=
                      sequence)
                return false;
            }
            binding.input_planes[0].span_bytes = input_size - 1;
            const auto calls = state.calls;
            return executor->Execute(binding).status ==
                       hal::StatusId::blocked &&
                   state.calls == calls;
          },
          "effects must run at capture geometry and match prepared reference "
          "resize at writer geometry without frame allocation");
    }
  return ok;
}

bool TestExactTransportReservations() {
  for (const bool resize : {false, true}) {
    RunnerState state;
    auto config = Config(&state);
    if (resize)
      config.plan.host_output =
          SizedImage(8, 4, 24, 96, hal::Mutability::write_only);
    config.plan.budget.persistent_bytes = resize ? 144 : 48;
    config.plan.budget.scratch_bytes =
        resize ? video::GetCpuPrimitiveTable()
                     .resize_rgb24_bilinear_plan_scratch_bytes(4, 2, 8, 4)
               : 0;
    config.plan.budget.resource_count = resize ? 4 : 2;
    open_video::PreparedOpenVideoCompatibilitySetupResult setup;
    auto executor = open_video::PreparedOpenVideoCompatibilityExecutor::Create(
        config, &setup);
    if (!Require(executor != nullptr, "exact capture and optional writer "
                                      "transport reservations must suffice"))
      return false;
    std::array<std::byte, kFrameBytes> input{}, unused{};
    auto binding = Binding(input, &unused, 1);
    binding.output_descriptor = config.plan.host_output;
    std::array<std::byte, 96> writer{};
    binding.output_planes[0] = {
        nullptr, writer.data(), config.plan.host_output.planes[0].size_bytes,
        config.plan.host_output.planes[0].row_stride_bytes};
    g_fail_allocations.store(true, std::memory_order_relaxed);
    const auto frame = executor->Execute(binding);
    g_fail_allocations.store(false, std::memory_order_relaxed);
    hal::ReasonId reason = hal::ReasonId::none;
    if (!Require(frame.output_ready &&
                     executor->CommitOutput(frame.completion, true, &reason) ==
                         hal::StatusId::ready,
                 "exact reserved transport must execute without acquisition"))
      return false;
  }
  return true;
}

bool TestMalformedGeometryAndReservationRollback() {
  bool ok = true;
  for (int bad = 0; bad < 26; ++bad) {
    ok &= ChildSucceeds(
        [bad] {
          RunnerState state;
          auto config = Config(&state);
          auto &input = config.plan.host_input;
          auto &output = config.plan.host_output;
          bool reservation = false;
          switch (bad) {
          case 0:
            input.width = 0;
            break;
          case 1:
            input.plane_count = 2;
            break;
          case 2:
            input.format = hal::PixelFormat::bgr_u8;
            break;
          case 3:
            output.channel_order = hal::ChannelOrder::bgr;
            break;
          case 4:
            input.planes[0].offset_bytes = 1;
            break;
          case 5:
            output.planes[0].row_stride_bytes = 1;
            break;
          case 6:
            input.planes[0].size_bytes = 1;
            break;
          case 7:
            output.width = output.planes[0].width =
                std::numeric_limits<std::uint32_t>::max();
            break;
          case 8:
            input.planes[0].row_stride_bytes =
                std::numeric_limits<std::uint64_t>::max();
            break;
          case 9:
            config.plan.intent.stage_count = hal::kMaxPreparedStages + 1;
            break;
          case 10:
            config.plan.intent.stage_count = 0;
            break;
          case 11:
            config.plan.budget.persistent_bytes = 47;
            reservation = true;
            break;
          case 12:
            config.plan.budget.resource_count = 1;
            reservation = true;
            break;
          case 13:
            config.plan.budget.in_flight_slots = 0;
            reservation = true;
            break;
          case 14:
            output = SizedImage(8, 4, 24, 96, hal::Mutability::write_only);
            config.plan.budget.scratch_bytes = 0;
            reservation = true;
            break;
          case 15:
            output = SizedImage(8, 4, 24, 96, hal::Mutability::write_only);
            config.plan.budget.persistent_bytes = 143;
            reservation = true;
            break;
          case 16:
            output = SizedImage(8, 4, 24, 96, hal::Mutability::write_only);
            config.plan.budget.scratch_bytes =
                video::GetCpuPrimitiveTable()
                    .resize_rgb24_bilinear_plan_scratch_bytes(4, 2, 8, 4) -
                1;
            reservation = true;
            break;
          case 17:
            output = SizedImage(8, 4, 24, 96, hal::Mutability::write_only);
            config.plan.budget.resource_count = 3;
            reservation = true;
            break;
          case 18:
            output.planes[0].size_bytes =
                std::numeric_limits<std::uint64_t>::max();
            break;
          case 19:
            input.mutability = hal::Mutability::write_only;
            break;
          case 20:
            output.mutability = hal::Mutability::read_only;
            break;
          case 21:
            output.orientation = hal::Orientation::bottom_left;
            break;
          case 22:
            input.planes[0].width = 3;
            break;
          case 23:
            output.planes[0].offset_bytes = 1;
            break;
          case 24:
            input = SizedImage(5, 3, 19, 52, hal::Mutability::read_only);
            break;
          case 25:
            input = SizedImage(5, 3, 19, 53, hal::Mutability::read_only);
            config.plan.budget.persistent_bytes = 137;
            reservation = true;
            break;
          }
          open_video::PreparedOpenVideoCompatibilitySetupResult setup;
          if (open_video::PreparedOpenVideoCompatibilityExecutor::Create(
                  config, &setup) ||
              setup.readiness.available || state.calls != 0 ||
              setup.status != (reservation ? hal::StatusId::resource_exhausted
                                           : hal::StatusId::blocked))
            return false;
          auto valid =
              open_video::PreparedOpenVideoCompatibilityExecutor::Create(
                  Config(&state), &setup);
          return valid && setup.readiness.available;
        },
        "malformed descriptors/stage count and insufficient reservations must "
        "reject before unsafe acquisition and permit clean retry");
  }
  return ok;
}

bool TestMixedOrMalformedAttributionIsRejected() {
  RunnerState state;
  open_video::PreparedOpenVideoCompatibilitySetupResult setup;

  auto malformed_domain = Config(&state);
  malformed_domain.selected_backend = hal::BackendId::cuda;
  bool ok = Require(
      !open_video::PreparedOpenVideoCompatibilityExecutor::Create(
          malformed_domain, &setup) &&
          setup.status == hal::StatusId::blocked &&
          setup.reason == hal::ReasonId::malformed_execution_domain,
      "selected backend must equal the exact execution-domain backend");

  auto mixed_backend = Config(&state);
  mixed_backend.backends[1] = hal::BackendId::cuda;
  ok &= Require(
      !open_video::PreparedOpenVideoCompatibilityExecutor::Create(
          mixed_backend, &setup) &&
          setup.status == hal::StatusId::blocked &&
          setup.reason == hal::ReasonId::semantic_provider_unavailable,
      "one-domain executor must reject a forged mixed backend array");

  auto mixed_provider = Config(&state);
  mixed_provider.providers[1] =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  ok &= Require(
      !open_video::PreparedOpenVideoCompatibilityExecutor::Create(
          mixed_provider, &setup) &&
          setup.status == hal::StatusId::blocked &&
          setup.reason == hal::ReasonId::semantic_provider_unavailable,
      "one-domain executor must reject a forged mixed provider array");

  auto provider_domain_mismatch = Config(&state);
  provider_domain_mismatch.selected_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  provider_domain_mismatch.providers[0] =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  provider_domain_mismatch.providers[1] =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  ok &= Require(
      !open_video::PreparedOpenVideoCompatibilityExecutor::Create(
          provider_domain_mismatch, &setup) &&
          setup.status == hal::StatusId::blocked &&
          setup.reason == hal::ReasonId::malformed_execution_domain,
      "provider residency must match the selected execution domain");
  return ok;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  bool ok = true;
  ok &= TestFailOpenPreservesAndExecuteDoesNotAllocate();
  ok &= TestFailedPublicationClearsCurrentAttributionOnly();
  ok &= TestExecutionFailureAndCleanupClearOnlyCurrentActivity();
  ok &= TestCameraWriterAcceptsProcessedOwnedOutput();
  ok &= TestTransportDomainAndCompletionOwner();
  ok &= TestGuardedCaptureUpscale();
  ok &= TestCaptureEffectsThenPreparedWriterResize();
  ok &= TestExactTransportReservations();
  ok &= TestMalformedGeometryAndReservationRollback();
  ok &= TestMixedOrMalformedAttributionIsRejected();
  if (!ok)
    return 1;
  std::cout << "Prepared Open Video compatibility executor tests passed\n";
  return 0;
}
