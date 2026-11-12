#include "core/video/cpu_primitive_hal.h"
#include "core/video/production_video_backend_manager.h"
#include "core/video/production_video_setup_request.h"
#include "core/vulkan/production_video_backend_setup_source.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> count_frame_allocations{false};
std::atomic<std::uint64_t> frame_allocations{0};
} // namespace
void *operator new(std::size_t size) {
  if (count_frame_allocations.load(std::memory_order_relaxed))
    ++frame_allocations;
  if (auto *memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

namespace {
namespace hal = studiocast::hal;
namespace video = studiocast::video;
struct Options {
  bool require = false;
  std::string model_id;
  std::filesystem::path model_root;
  std::string replacement;
  unsigned frames = 3;
};
bool Never(void *) noexcept { return false; }
std::uint64_t Now(void *) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
void Poll(void *, std::uint64_t delay) noexcept {
  std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
}
bool Run(const Options &options, video::effects::VirtualBackgroundMode mode,
         bool keylight, bool final) {
  video::effects::BroadcastCameraEffects effects;
  effects.virtual_background.mode = mode;
  effects.virtual_background.model_id = options.model_id;
  effects.virtual_background.replace_path = options.replacement;
  effects.virtual_background.remove_color = "#173e91";
  effects.virtual_background.strength = 17;
  effects.virtual_key_light.enabled = keylight;
  effects.virtual_key_light.intensity = 65;
  effects.virtual_key_light.temperature_preset = 1;
  effects.virtual_key_light.direction_pan_degrees = -35;
  effects.vignette.enabled = final;
  effects.vignette.center_on_tracked_face = false;
  effects.vignette.intensity = 47;
  effects.mirror = final;
  int context = 0;
  video::ProductionVideoBoundedWaitPolicy wait;
  wait.timeout_nanoseconds = 100000000;
  wait.poll_interval_nanoseconds = 1000000;
  wait.maximum_poll_count = 100;
  wait.stop_requested = Never;
  wait.cancel_requested = Never;
  wait.now_nanoseconds = Now;
  wait.poll_wait = Poll;
  wait.callback_context = &context;
  wait.callback_context_generation = 1;
  wait.callback_context_lifetime = hal::ResourceLifetime::adapter_context;
  const auto request = video::BuildProductionVideoBackendSetupRequest(
      effects, video::ComputeBackendPreference::vulkan, 64, 48, 32, 24,
      video::PixelFormat::rgb24, 1, 1, 1, wait, video::GetCpuPrimitiveTable());
  video::ProductionVideoBackendManager manager(
      studiocast::vulkan::CreateProductionVideoBackendSetupSource());
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto status = manager.Setup(request, &evidence, &reason);
  const auto setup = manager.Snapshot();
  std::cout << "case background=" << static_cast<unsigned>(mode)
            << " keylight=" << keylight << " final=" << final
            << " setup_status=" << static_cast<unsigned>(status)
            << " reason=" << static_cast<unsigned>(reason)
            << " candidate_reason="
            << static_cast<unsigned>(evidence.candidate_readiness.reason)
            << '\n';
  if (status != hal::StatusId::ready || !setup.resident_prepared ||
      setup.readiness.selected_backend != hal::BackendId::vulkan ||
      setup.resident.plan.prepared_program.stage_count !=
          request.resident.plan.intent.stage_count)
    return false;
  const auto &program = setup.resident.plan.prepared_program;
  for (std::size_t i = 0; i < program.stage_count; ++i) {
    const auto &stage = program.stages[i];
    if (stage.semantic != hal::SemanticEffectId::mirror &&
        stage.semantic != hal::SemanticEffectId::vignette &&
        (stage.model.artifact_generation == 0 ||
         std::all_of(stage.model.artifact_set_hash.begin(),
                     stage.model.artifact_set_hash.end(),
                     [](auto value) { return value == 0; })))
      return false;
  }
  std::vector<std::byte> input(64 * 48 * 3), output(32 * 24 * 3);
  std::uint64_t output_hash = 0;
  for (unsigned f = 0; f < options.frames; ++f) {
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<std::byte>((i * 13 + f * 31) & 255);
    hal::HostFrameBinding binding;
    binding.input_descriptor = request.resident.plan.host_input;
    binding.output_descriptor = request.resident.plan.host_output;
    binding.input_plane_count = binding.output_plane_count = 1;
    binding.input_planes[0].read_data = input.data();
    binding.input_planes[0].span_bytes = input.size();
    binding.input_planes[0].row_stride_bytes = 64 * 3;
    binding.output_planes[0].write_data = output.data();
    binding.output_planes[0].span_bytes = output.size();
    binding.output_planes[0].row_stride_bytes = 32 * 3;
    binding.ownership = hal::OwnershipModel::borrowed_until_completion;
    binding.lifetime = hal::ResourceLifetime::frame_borrowed;
    binding.capture_sequence = f + 1;
    frame_allocations.store(0, std::memory_order_relaxed);
    count_frame_allocations.store(true, std::memory_order_relaxed);
    const auto frame = manager.Execute(binding);
    count_frame_allocations.store(false, std::memory_order_relaxed);
    const auto allocations = frame_allocations.load(std::memory_order_relaxed);
    std::cout << "frame=" << f + 1
              << " status=" << static_cast<unsigned>(frame.status)
              << " reason=" << static_cast<unsigned>(frame.reason)
              << " stages=" << frame.stage_count
              << " uploads=" << frame.transfers.uploads
              << " final=" << frame.transfers.final_readbacks
              << " intermediate=" << frame.transfers.intermediate_readbacks
              << " tails=" << frame.transfers.cpu_tail_boundaries
              << " sync=" << frame.transfers.synchronizations
              << " host_allocations=" << allocations
              << " elapsed_ns=" << frame.frame_elapsed_ns << '\n';
    if (allocations != 0 || frame.status != hal::StatusId::executed ||
        !frame.output_ready || frame.stage_count != program.stage_count ||
        frame.transfers.uploads != 1 || frame.transfers.final_readbacks != 1 ||
        frame.transfers.intermediate_readbacks != 0 ||
        frame.transfers.intermediate_uploads != 0 ||
        frame.transfers.cpu_tail_boundaries != 0)
      return false;
    for (std::size_t i = 0; i < frame.stage_count; ++i)
      if (frame.stages[i].status != hal::StageExecutionStatus::succeeded)
        return false;
    if (manager.Snapshot().resident.latest_committed_frame.capture_sequence ==
        f + 1)
      return false;
    if (manager.CommitOutput(frame.completion, true, &reason) !=
        hal::StatusId::ready)
      return false;
    const auto committed = manager.Snapshot().resident.latest_committed_frame;
    if (!committed.valid || committed.capture_sequence != f + 1)
      return false;
    for (const auto byte : output)
      output_hash = (output_hash ^ static_cast<std::uint8_t>(byte)) *
                    UINT64_C(1099511628211);
  }
  const auto snapshot = manager.Snapshot();
  const auto &telemetry = snapshot.resident.telemetry;
  std::cout << "summary writer=in_memory_not_v4l2 committed="
            << telemetry.committed_output_frames
            << " fallback=" << telemetry.fallback_frames
            << " output_hash=" << output_hash << '\n';
  return telemetry.committed_output_frames == options.frames &&
         telemetry.fallback_frames == 0;
}
} // namespace
int main(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--require") {
      options.require = true;
      continue;
    }
    if (i + 1 >= argc)
      return 2;
    const std::string value = argv[++i];
    if (arg == "--model-root")
      options.model_root = value;
    else if (arg == "--model-id")
      options.model_id = value;
    else if (arg == "--replace-path")
      options.replacement = value;
    else if (arg == "--frames") {
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), options.frames);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || options.frames == 0 ||
          options.frames > 8)
        return 2;
    } else
      return 2;
  }
  if (!options.require) {
    std::cout << "[SKIP] explicit --require and preinstalled ncnn model are "
                 "required\n";
    return 77;
  }
  const auto root = options.model_root.lexically_normal();
  if (options.model_id.empty() || options.replacement.empty() ||
      root.filename() != "open_video" ||
      root.parent_path().filename() != "models" ||
      root.parent_path().parent_path().filename() != "studiocast") {
    std::cerr
        << "FAIL required model root must end in studiocast/models/open_video; "
           "model ID and replacement image are required\n";
    return 2;
  }
  if (setenv("XDG_DATA_HOME",
             root.parent_path().parent_path().parent_path().c_str(), 1) != 0)
    return 2;
  for (auto mode : {video::effects::VirtualBackgroundMode::blur,
                    video::effects::VirtualBackgroundMode::remove,
                    video::effects::VirtualBackgroundMode::replace}) {
    if (!Run(options, mode, false, false) || !Run(options, mode, true, true))
      return 1;
  }
  if (!Run(options, video::effects::VirtualBackgroundMode::none, true, false) ||
      !Run(options, video::effects::VirtualBackgroundMode::none, true, true))
    return 1;
  std::cout << "Vulkan model-backed production manager matrix passed\n";
  return 0;
}
