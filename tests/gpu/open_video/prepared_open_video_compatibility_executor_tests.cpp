#include "core/open_video/prepared_open_video_compatibility_executor.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

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

struct RunnerState {
  std::uint64_t calls = 0;
};

bool RunStage(void *opaque, const video::CanonicalVideoStage &stage,
              std::uint64_t, std::uint8_t *rgb, int, int, std::size_t,
              std::string *error) noexcept {
  auto *state = static_cast<RunnerState *>(opaque);
  ++state->calls;
  if (stage.intent.semantic == hal::SemanticEffectId::video_noise_removal) {
    rgb[0] = 0x22;
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
  if (!TestFailOpenPreservesAndExecuteDoesNotAllocate())
    return 1;
  std::cout << "Prepared Open Video compatibility executor tests passed\n";
  return 0;
}
