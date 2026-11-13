#include "core/open_video/model_pack_registry.h"
#include "core/video/cpu_primitive_hal.h"
#include "core/video/image_ppm.h"
#include "core/video/mjpeg_decode.h"
#include "core/video/production_video_backend_manager.h"
#include "core/video/production_video_setup_request.h"
#include "core/vulkan/production_video_backend_setup_source.h"
#include "core/vulkan/video_execution_factory.h"
#include "core/vulkan/video_execution_factory_internal.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> count_allocations{false};
std::atomic<std::uint64_t> allocations{0};
void CountAllocation() noexcept {
  if (count_allocations.load(std::memory_order_relaxed))
    allocations.fetch_add(1, std::memory_order_relaxed);
}
} // namespace
void *operator new(std::size_t size) {
  CountAllocation();
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
void *operator new(std::size_t size, std::align_val_t alignment) {
  CountAllocation();
  void *memory = nullptr;
  if (posix_memalign(&memory, static_cast<std::size_t>(alignment),
                     size ? size : 1) != 0)
    throw std::bad_alloc();
  return memory;
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void operator delete(void *memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete(void *memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}

namespace {
namespace hal = studiocast::hal;
namespace video = studiocast::video;
namespace vulkan = studiocast::vulkan;
using Effects = video::effects::BroadcastCameraEffects;
using Background = video::effects::VirtualBackgroundMode;
constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_vulkan_production_kernels;
constexpr auto kMissing = "studiocast-smoke-explicitly-missing-model";

struct Options {
  bool require = false;
  bool prepare_only = false;
  std::filesystem::path model_root;
  std::string model_id = "modnet-webnn-256-fp32";
  std::string face_model_id = "yunet_opencv_zoo_2023mar_fp32";
  std::string eye_model_id = "gaze_correction_cam_flx_v0_1_1";
  std::string denoise_model_id = "fastdvdnet_sigma15";
  std::filesystem::path face_path;
  std::string face_sha256;
  std::string replacement;
  std::string case_name;
  unsigned frames = 4;
  unsigned width = 0, height = 0, output_width = 0, output_height = 0;
};

bool Require(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "FAIL " << message << '\n';
  return condition;
}
bool Number(std::string_view text, unsigned *out) {
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}
bool Geometry(std::string_view text, unsigned *width, unsigned *height) {
  const auto split = text.find('x');
  return split != std::string_view::npos &&
         Number(text.substr(0, split), width) &&
         Number(text.substr(split + 1), height) && *width > 0 && *height > 0 &&
         *width <= 4096 && *height <= 4096;
}
bool Parse(int argc, char **argv, Options *options) {
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--require") {
      options->require = true;
      continue;
    }
    if (arg == "--prepare-only") {
      options->prepare_only = true;
      continue;
    }
    if (index + 1 == argc)
      return false;
    const std::string value = argv[++index];
    if (arg == "--model-root")
      options->model_root = value;
    else if (arg == "--model-id")
      options->model_id = value;
    else if (arg == "--face-model-id")
      options->face_model_id = value;
    else if (arg == "--eye-model-id")
      options->eye_model_id = value;
    else if (arg == "--denoise-model-id")
      options->denoise_model_id = value;
    else if (arg == "--face-path")
      options->face_path = value;
    else if (arg == "--face-sha256")
      options->face_sha256 = value;
    else if (arg == "--replace-path")
      options->replacement = value;
    else if (arg == "--case")
      options->case_name = value;
    else if (arg == "--input") {
      if (!Geometry(value, &options->width, &options->height))
        return false;
    } else if (arg == "--output") {
      if (!Geometry(value, &options->output_width, &options->output_height))
        return false;
    } else if (arg == "--frames") {
      if (!Number(value, &options->frames) || options->frames < 3 ||
          options->frames > 8)
        return false;
    } else
      return false;
  }
  return true;
}

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
int wait_context = 0;
video::ProductionVideoBoundedWaitPolicy Wait() {
  video::ProductionVideoBoundedWaitPolicy wait;
  wait.timeout_nanoseconds = UINT64_C(100000000);
  wait.poll_interval_nanoseconds = UINT64_C(1000000);
  wait.maximum_poll_count = 100;
  wait.stop_requested = Never;
  wait.cancel_requested = Never;
  wait.now_nanoseconds = Now;
  wait.poll_wait = Poll;
  wait.callback_context = &wait_context;
  wait.callback_context_generation = 1;
  wait.callback_context_lifetime = hal::ResourceLifetime::adapter_context;
  return wait;
}

// Observe the real setup result; all model discovery and native construction
// still belong to the production source. The copied evidence owns no GPU data.
class ObservedSource final : public video::IProductionVideoBackendSetupSource {
public:
  std::unique_ptr<video::IProductionVideoBackendSetupSource> source =
      vulkan::CreateProductionVideoBackendSetupSource();
  video::ProductionVideoSetupLeaseSnapshot lease{};
  std::array<video::ProductionVideoResidentSetupSourceResult::ModelSelection,
             hal::kMaxPreparedStages>
      models{};
  std::size_t model_count = 0;
  unsigned calls = 0;
  video::ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    ++calls;
    auto result = source->AcquireResidentSetup(request);
    lease = result.lease.snapshot();
    models = result.model_selections;
    model_count = result.model_selection_count;
    return result;
  }
};

std::string Hex(const std::array<std::uint8_t, 32> &bytes) {
  std::ostringstream out;
  for (const auto byte : bytes)
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned>(byte);
  return out.str();
}
std::uint64_t Checksum(const std::vector<std::uint8_t> &bytes) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const auto byte : bytes)
    hash = (hash ^ byte) * UINT64_C(1099511628211);
  return hash;
}

struct Case {
  std::string name;
  Effects effects;
  unsigned width = 64, height = 48;
  unsigned output_width = 32, output_height = 24;
  bool face = false;
  bool exact_preservation = false;
};
Effects Base(const Options &options) {
  Effects effects;
  effects.virtual_background.model_id = options.model_id;
  effects.virtual_background.replace_path = options.replacement;
  effects.virtual_background.remove_color = "#173e91";
  effects.virtual_background.strength = 17;
  effects.virtual_key_light.intensity = 65;
  effects.virtual_key_light.direction_pan_degrees = -35;
  effects.auto_frame.model_id = options.face_model_id;
  effects.auto_frame.strength = 65;
  effects.auto_frame.smoothing = 35;
  effects.eye_contact.model_id = options.eye_model_id;
  effects.eye_contact.strength = 100;
  effects.video_noise_removal.model_id = options.denoise_model_id;
  effects.video_noise_removal.strength = 70;
  effects.vignette.intensity = 47;
  effects.vignette.center_on_tracked_face = false;
  return effects;
}
std::vector<Case> Matrix(const Options &options) {
  std::vector<Case> cases;
  const auto add = [&](std::string name, Effects effects, bool face = false,
                       unsigned width = 64, unsigned height = 48,
                       bool preserve = false) {
    Case test{std::move(name), std::move(effects), width, height,
              width / 2,       height / 2,         face,  preserve};
    if (preserve) {
      test.output_width = width;
      test.output_height = height;
    }
    cases.push_back(std::move(test));
  };
  auto effects = Base(options);
  effects.video_noise_removal.enabled = true;
  add("denoise", effects);
  effects.video_noise_removal.strength = 0;
  add("denoise_zero", effects, false, 65, 49, true);
  effects = Base(options);
  effects.eye_contact.enabled = true;
  add("eye_no_face", effects, false, 64, 48, true);
  effects.eye_contact.strength = 0;
  add("eye_zero", effects, false, 64, 48, true);
  effects = Base(options);
  effects.auto_frame.enabled = true;
  add("auto_frame", effects);
  effects.vignette.enabled = true;
  effects.vignette.center_on_tracked_face = true;
  add("tracked_vignette", effects);
  effects = Base(options);
  effects.virtual_key_light.enabled = true;
  add("keylight", effects);
  for (const auto mode :
       {Background::blur, Background::remove, Background::replace}) {
    effects = Base(options);
    effects.virtual_background.mode = mode;
    const std::string mode_name = mode == Background::blur     ? "blur"
                                  : mode == Background::remove ? "remove"
                                                               : "replace";
    add(mode_name, effects);
    effects.video_noise_removal.enabled = effects.eye_contact.enabled = true;
    effects.virtual_key_light.enabled = effects.auto_frame.enabled = true;
    effects.vignette.enabled = effects.mirror = true;
    effects.vignette.center_on_tracked_face = true;
    add("all_" + mode_name, effects);
    add("all_" + mode_name + "_odd", effects, false, 65, 49);
    if (!options.face_path.empty())
      add("all_" + mode_name + "_face", effects, true, 320, 240);
  }
  effects = Base(options);
  effects.video_noise_removal.enabled = effects.eye_contact.enabled = true;
  add("denoise_eye", effects);
  effects.auto_frame.enabled = true;
  add("denoise_eye_auto", effects);
  if (!options.face_path.empty()) {
    effects = Base(options);
    effects.eye_contact.enabled = true;
    add("eye_face", effects, true, 320, 240, true);
    effects.eye_contact.strength = 0;
    add("eye_face_zero", effects, true, 320, 240, true);
  }
  for (auto &test : cases) {
    if (options.width) {
      test.width = options.width;
      test.height = options.height;
    }
    if (options.output_width) {
      test.output_width = options.output_width;
      test.output_height = options.output_height;
    } else if (options.width) {
      test.output_width = test.exact_preservation ? test.width : test.width / 2;
      test.output_height =
          test.exact_preservation ? test.height : test.height / 2;
    }
  }
  return cases;
}

bool LoadFace(const Options &options, video::Rgb24Frame *face) {
  if (options.face_path.empty())
    return true;
  const auto hash =
      options.face_sha256.empty()
          ? "ede6804edf0ec005144ffe7f0f2b56bd2aa1d21beee79d29f72cb16ead19a447"
          : options.face_sha256;
  std::string error;
  auto snapshot = studiocast::open_video::OpenVerifiedModelFileSnapshot(
      options.face_path, hash, 8u * 1024 * 1024, &error);
  if (!Require(static_cast<bool>(snapshot),
               "local face fixture checksum/snapshot failed: " + error))
    return false;
  std::vector<std::uint8_t> jpeg(8u * 1024 * 1024);
  const auto size = std::fread(jpeg.data(), 1, jpeg.size(), snapshot.get());
  int width = 0, height = 0;
  if (!Require(size > 0 && video::DecodeMjpegToRgb24(jpeg.data(), size, *face,
                                                     width, height, &error),
               "local face fixture JPEG decode failed: " + error))
    return false;
  std::cout << "local_face_fixture_sha256=" << hash << " decoded=" << width
            << 'x' << height << " redistribution=none\n";
  return true;
}

video::ProductionVideoBackendSetupRequest Request(const Case &test) {
  return video::BuildProductionVideoBackendSetupRequest(
      test.effects, video::ComputeBackendPreference::vulkan,
      static_cast<int>(test.width), static_cast<int>(test.height),
      static_cast<int>(test.output_width), static_cast<int>(test.output_height),
      video::PixelFormat::rgb24, 1, 1, 1, Wait(),
      video::GetCpuPrimitiveTable());
}
bool SameBudget(const hal::ResourceBudget &a, const hal::ResourceBudget &b) {
  return a.persistent_bytes == b.persistent_bytes &&
         a.scratch_bytes == b.scratch_bytes &&
         a.upload_staging_bytes == b.upload_staging_bytes &&
         a.readback_staging_bytes == b.readback_staging_bytes &&
         a.resource_count == b.resource_count &&
         a.in_flight_slots == b.in_flight_slots;
}
hal::HostFrameBinding
Binding(const video::ProductionVideoBackendSetupRequest &request,
        const std::vector<std::uint8_t> &input,
        std::vector<std::uint8_t> *output, std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = request.resident.plan.host_input;
  binding.output_descriptor = request.resident.plan.host_output;
  binding.input_plane_count = binding.output_plane_count = 1;
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input.data());
  binding.input_planes[0].span_bytes = input.size();
  binding.input_planes[0].row_stride_bytes =
      request.resident.plan.host_input.width * 3u;
  binding.output_planes[0].write_data =
      reinterpret_cast<std::byte *>(output->data());
  binding.output_planes[0].span_bytes = output->size();
  binding.output_planes[0].row_stride_bytes =
      request.resident.plan.host_output.width * 3u;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  binding.capture_sequence = sequence;
  return binding;
}

bool Collect(const ObservedSource &source,
             vulkan::VulkanProductionModelDiagnostics *out) {
  const auto *factory =
      dynamic_cast<const vulkan::VulkanVideoExecutionFactory *>(
          source.lease.factory.factory);
  std::string error;
  return Require(factory && vulkan::CollectProductionVulkanModelDiagnostics(
                                *factory, out, &error),
                 "actual factory diagnostics unavailable: " + error);
}
bool RuntimeSafe(
    const studiocast::open_vulkan::VulkanMattingRuntimeEvidence &runtime,
    const hal::ExecutionDomainId &domain) {
  if (!runtime.graph_loaded)
    return true;
  return runtime.runtime_created && runtime.warmup_complete &&
         !runtime.cpu_layers_used && runtime.device_identity_matches &&
         runtime.shared_device_imported && runtime.queue_ownership_explicit &&
         runtime.synchronous_completion &&
         runtime.bounded_reusable_allocations &&
         runtime.input_device_resident && runtime.output_device_resident &&
         runtime.dynamic_allocation_count == 0 &&
         runtime.cpu_readback_count == 0 &&
         runtime.active_device.context_generation ==
             domain.logical_context_generation &&
         runtime.active_device.vendor_id == domain.physical_device.vendor_id &&
         runtime.active_device.device_id == domain.physical_device.device_id;
}
bool NativeEvidence(const vulkan::VulkanProductionModelDiagnostics &before,
                    const vulkan::VulkanProductionModelDiagnostics &after,
                    const Case &test, const hal::ExecutionDomainId &domain,
                    std::uint64_t frames) {
  if (!Require(after.allocations.current_bytes ==
                       before.allocations.current_bytes &&
                   after.allocations.allocation_count ==
                       before.allocations.allocation_count &&
                   after.allocations.high_water_bytes ==
                       before.allocations.high_water_bytes &&
                   after.allocations.high_water_allocation_count ==
                       before.allocations.high_water_allocation_count &&
                   after.allocations.current_bytes <=
                       after.allocations.budget_bytes &&
                   after.allocations.allocation_count <=
                       after.allocations.resource_limit &&
                   after.completed_batches == before.completed_batches + frames,
               "native device allocation reuse, quota, or one-batch completion "
               "evidence failed"))
    return false;
  const auto runtime_check = [&](const auto &initial, const auto &actual,
                                 bool runs) {
    return RuntimeSafe(actual, domain) &&
           actual.warmup_inference_count == initial.warmup_inference_count &&
           actual.inference_count ==
               initial.inference_count + (runs ? frames : 0);
  };
  const bool face = test.effects.auto_frame.enabled ||
                    (test.effects.eye_contact.enabled &&
                     test.effects.eye_contact.strength > 0);
  const bool matte = test.effects.virtual_background.mode != Background::none ||
                     test.effects.virtual_key_light.enabled ||
                     test.effects.auto_frame.enabled;
  const bool eye =
      test.effects.eye_contact.enabled && test.effects.eye_contact.strength > 0;
  return Require(
      runtime_check(before.matting, after.matting, matte) &&
          runtime_check(before.denoise, after.denoise,
                        test.effects.video_noise_removal.enabled &&
                            test.effects.video_noise_removal.strength > 0) &&
          runtime_check(before.face, after.face, face) &&
          runtime_check(before.left_eye, after.left_eye, eye) &&
          runtime_check(before.right_eye, after.right_eye, eye),
      "native graph execution/shared current-frame inference/residency "
      "evidence failed");
}

bool RunCase(const Options &options, const Case &test,
             const video::Rgb24Frame &face,
             std::vector<std::uint8_t> *last_output = nullptr) {
  const auto request = Request(test);
  auto source = std::make_unique<ObservedSource>();
  auto *observed = source.get();
  video::ProductionVideoBackendManager manager(std::move(source));
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto setup_begin = Now(nullptr);
  const auto status = manager.Setup(request, &evidence, &reason);
  const auto setup = manager.Snapshot();
  const auto &program = setup.resident.plan.prepared_program;
  std::cout << "case=" << test.name
            << " setup=" << static_cast<unsigned>(status)
            << " reason=" << static_cast<unsigned>(reason)
            << " setup_ns=" << Now(nullptr) - setup_begin
            << " input=" << test.width << 'x' << test.height
            << " output=" << test.output_width << 'x' << test.output_height
            << " wait_ns=" << request.resident.wait.timeout_nanoseconds
            << " stages=" << program.stage_count << '\n';
  if (!Require(status == hal::StatusId::ready && setup.resident_prepared &&
                   setup.readiness.selected_backend == hal::BackendId::vulkan &&
                   setup.readiness.provider == kProvider &&
                   program.stage_count ==
                       request.resident.plan.intent.stage_count &&
                   program.cpu_tail_count == 0 &&
                   program.maximum_metadata_readbacks_per_frame == 0 &&
                   program.maximum_metadata_cpu_tails_per_frame == 0 &&
                   SameBudget(program.budget, request.resident.plan.budget) &&
                   SameBudget(observed->lease.reserved_budget,
                              request.resident.plan.budget),
               "exact fully resident Vulkan setup/budget contract failed"))
    return false;
  const auto &domain = setup.readiness.execution_domain;
  std::cout << "domain vendor=" << domain.physical_device.vendor_id
            << " device=" << domain.physical_device.device_id
            << " context=" << domain.logical_context_generation
            << " queue=" << domain.queue_or_stream_identity
            << " adapter=" << domain.adapter_instance_id
            << " reserved_persistent=" << program.budget.persistent_bytes
            << " reserved_scratch=" << program.budget.scratch_bytes
            << " reserved_resources=" << program.budget.resource_count << '\n';
  const hal::ModelArtifactIdentity *matte = nullptr;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (!Require(stage.semantic == request.resident.plan.intent.stages[index]
                                       .intent.semantic &&
                     stage.backend == hal::BackendId::vulkan &&
                     stage.provider == kProvider &&
                     stage.execution_domain == domain,
                 "canonical order or exact per-stage domain/provider changed"))
      return false;
    const bool model = stage.semantic != hal::SemanticEffectId::mirror &&
                       stage.semantic != hal::SemanticEffectId::vignette;
    if (model && !Require(stage.model.artifact_generation != 0 &&
                              std::any_of(stage.model.artifact_set_hash.begin(),
                                          stage.model.artifact_set_hash.end(),
                                          [](auto byte) { return byte != 0; }),
                          "model stage lacks a verified artifact identity"))
      return false;
    const bool matting =
        stage.semantic == hal::SemanticEffectId::virtual_background_blur ||
        stage.semantic == hal::SemanticEffectId::virtual_background_remove ||
        stage.semantic == hal::SemanticEffectId::virtual_background_replace ||
        stage.semantic == hal::SemanticEffectId::virtual_key_light;
    if (matting) {
      if (matte &&
          !Require(matte->artifact_set_hash == stage.model.artifact_set_hash &&
                       matte->artifact_generation ==
                           stage.model.artifact_generation,
                   "background and key light selected different current-frame "
                   "matte identities"))
        return false;
      matte = &stage.model;
    }
    std::cout << "stage=" << index
              << " semantic=" << static_cast<unsigned>(stage.semantic)
              << " artifact_generation=" << stage.model.artifact_generation
              << " artifact_set_sha256=" << Hex(stage.model.artifact_set_hash)
              << '\n';
  }
  for (std::size_t index = 0; index < observed->model_count; ++index)
    std::cout << "selection semantic="
              << static_cast<unsigned>(observed->models[index].semantic)
              << " requested=" << observed->models[index].requested_model_id
              << " selected=" << observed->models[index].selected_model_id
              << '\n';
  vulkan::VulkanProductionModelDiagnostics native_before;
  if (!Collect(*observed, &native_before))
    return false;
  std::cout << "native_prepared_bytes="
            << native_before.allocations.current_bytes
            << " native_prepared_allocations="
            << native_before.allocations.allocation_count
            << " native_high_water_bytes="
            << native_before.allocations.high_water_bytes
            << " native_high_water_allocations="
            << native_before.allocations.high_water_allocation_count << '\n';
  if (options.prepare_only) {
    const auto shutdown = manager.Shutdown();
    return Require(shutdown.quiescent &&
                       shutdown.driver_context_release_permitted &&
                       !shutdown.pending,
                   "prepared-only shutdown did not prove quiescence");
  }
  std::vector<std::uint8_t> input(static_cast<std::size_t>(test.width) *
                                  test.height * 3);
  std::vector<std::uint8_t> output(static_cast<std::size_t>(test.output_width) *
                                   test.output_height * 3);
  if (test.face) {
    std::string error;
    if (!Require(video::ResizeRgb24Bilinear(face.data(), face.width,
                                            face.height, face.stride_bytes,
                                            static_cast<int>(test.width),
                                            static_cast<int>(test.height),
                                            &input, test.width * 3, &error),
                 error))
      return false;
  } else if (test.name != "eye_no_face") {
    for (std::size_t index = 0; index < input.size(); ++index)
      input[index] = static_cast<std::uint8_t>((index * 13 + 21) & 255);
  }
  unsigned committed = 0;
  std::uint64_t executed = 0;
  const auto execute = [&](std::uint64_t sequence, bool publish) {
    const auto binding = Binding(request, input, &output, sequence);
    const auto before = manager.Snapshot().resident.latest_committed_frame;
    allocations.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    const auto frame = manager.Execute(binding);
    count_allocations.store(false, std::memory_order_relaxed);
    const auto allocated = allocations.load(std::memory_order_relaxed);
    std::cout << "frame case=" << test.name << " sequence=" << sequence
              << " status=" << static_cast<unsigned>(frame.status)
              << " reason=" << static_cast<unsigned>(frame.reason)
              << " upload=" << frame.transfers.uploads
              << " final=" << frame.transfers.final_readbacks
              << " intermediate=" << frame.transfers.intermediate_readbacks
              << " tails=" << frame.transfers.cpu_tail_boundaries
              << " sync=" << frame.transfers.synchronizations
              << " cpp_heap_allocations=" << allocated
              << " elapsed_ns=" << frame.frame_elapsed_ns << '\n';
    if (!Require(frame.status == hal::StatusId::executed &&
                     frame.output_ready && allocated == 0 &&
                     frame.stage_count == program.stage_count &&
                     frame.execution_domain == domain &&
                     frame.transfers.uploads == 1 &&
                     frame.transfers.final_readbacks == 1 &&
                     frame.transfers.intermediate_uploads == 0 &&
                     frame.transfers.intermediate_readbacks == 0 &&
                     frame.transfers.cpu_tail_boundaries == 0 &&
                     frame.transfers.synchronizations == 1,
                 "frame did not execute one bounded resident batch with "
                 "final-only transport"))
      return false;
    ++executed;
    vulkan::VulkanProductionModelDiagnostics native_after;
    if (!Collect(*observed, &native_after) ||
        !NativeEvidence(native_before, native_after, test, domain, executed))
      return false;
    for (std::size_t index = 0; index < frame.stage_count; ++index)
      if (!Require(frame.stages[index].status ==
                           hal::StageExecutionStatus::succeeded &&
                       frame.stages[index].semantic ==
                           program.stages[index].semantic,
                   "enabled prepared stage did not succeed in canonical order"))
        return false;
    const auto pending = manager.Snapshot().resident.latest_committed_frame;
    if (!Require(!pending.valid || pending.capture_sequence != sequence,
                 "unwritten output was attributed as active"))
      return false;
    const auto commit =
        manager.CommitOutput(frame.completion, publish, &reason);
    const auto after = manager.Snapshot().resident.latest_committed_frame;
    if (publish) {
      ++committed;
      if (!Require(commit == hal::StatusId::ready && after.valid &&
                       after.capture_sequence == sequence &&
                       after.execution_domain == domain &&
                       after.stage_count == program.stage_count &&
                       (!before.valid || after.output_commit_sequence >
                                             before.output_commit_sequence),
                   "writer commit did not publish this exact "
                   "domain/stage/frame identity"))
        return false;
    } else if (!Require(
                   commit == hal::StatusId::failed && !after.valid,
                   "failed writer publication left stale active attribution"))
      return false;
    if (test.exact_preservation &&
        (!test.face || test.effects.eye_contact.strength == 0) &&
        !Require(output == input,
                 "zero-strength/no-face path did not preserve exact input"))
      return false;
    return true;
  };
  std::vector<std::uint8_t> gap_input, gap_output;
  for (unsigned frame = 0; frame < options.frames; ++frame) {
    if (test.name == "denoise")
      for (std::size_t index = 0; index < input.size(); ++index)
        input[index] =
            static_cast<std::uint8_t>((index * 13 + frame * 47 + 21) & 255);
    if (frame == 1 &&
        !Require(manager.ResetTemporal(hal::ReasonId::temporal_discontinuity) ==
                     hal::StatusId::ready,
                 "explicit temporal reset failed"))
      return false;
    const std::uint64_t sequence = frame + 1u + (frame >= 2 ? 4u : 0u);
    if (!execute(sequence, true))
      return false;
    if (frame == 2 && test.name == "denoise") {
      gap_input = input;
      gap_output = output;
    }
  }
  if (test.name == "denoise") {
    input = gap_input;
    if (!Require(manager.ResetTemporal(hal::ReasonId::temporal_discontinuity) ==
                     hal::StatusId::ready,
                 "denoise reset oracle could not clear old history") ||
        !execute(options.frames + 5u, true) ||
        !Require(output == gap_output, "capture-sequence gap did not match "
                                       "explicit-reset denoise history"))
      return false;
  }
  if (!execute(options.frames + 6u, false) ||
      !execute(options.frames + 7u, true))
    return false;
  const auto final = manager.Snapshot();
  const auto &telemetry = final.resident.telemetry;
  if (!Require(observed->calls == 1 &&
                   telemetry.committed_output_frames == committed &&
                   telemetry.active_backend_frames == committed &&
                   telemetry.fallback_frames == 0 &&
                   telemetry.optional_fail_open_stages == 0 &&
                   telemetry.execution_attempts == executed &&
                   telemetry.execution_successes == executed &&
                   telemetry.transfers.uploads == executed &&
                   telemetry.transfers.final_readbacks == executed &&
                   telemetry.transfers.synchronizations == executed &&
                   !final.resident.quarantined &&
                   telemetry.temporal_resets >= 1,
               "steady-state lifecycle performed discovery, fallback, or stale "
               "attribution"))
    return false;
  std::cout << "summary case=" << test.name << " committed=" << committed
            << " resets=" << telemetry.temporal_resets
            << " rgb_fnv1a64=" << Checksum(output)
            << " writer=in_memory_not_v4l2\n";
  if (last_output)
    *last_output = output;
  const auto shutdown = manager.Shutdown();
  return Require(shutdown.quiescent &&
                     shutdown.driver_context_release_permitted &&
                     !shutdown.pending,
                 "shutdown did not prove complete-island quiescence");
}

bool SetupIsolation(const Options &options) {
  for (const auto semantic :
       {hal::SemanticEffectId::video_noise_removal,
        hal::SemanticEffectId::eye_contact, hal::SemanticEffectId::auto_frame,
        hal::SemanticEffectId::virtual_background_blur}) {
    auto effects = Base(options);
    effects.mirror = true;
    if (semantic == hal::SemanticEffectId::video_noise_removal) {
      effects.video_noise_removal.enabled = true;
      effects.video_noise_removal.model_id = kMissing;
    } else if (semantic == hal::SemanticEffectId::eye_contact) {
      effects.eye_contact.enabled = true;
      effects.eye_contact.model_id = kMissing;
    } else if (semantic == hal::SemanticEffectId::auto_frame) {
      effects.auto_frame.enabled = true;
      effects.auto_frame.model_id = kMissing;
    } else {
      effects.virtual_background.mode = Background::blur;
      effects.virtual_background.model_id = kMissing;
    }
    Case test{"optional_missing_pack", effects};
    const auto request = Request(test);
    video::ProductionVideoBackendManager manager(
        vulkan::CreateProductionVideoBackendSetupSource());
    video::ProductionVideoBackendSetupEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status = manager.Setup(request, &evidence, &reason);
    const auto setup = manager.Snapshot();
    std::cout << "isolation semantic=" << static_cast<unsigned>(semantic)
              << " setup=" << static_cast<unsigned>(status)
              << " reason=" << static_cast<unsigned>(reason) << '\n';
    if (!Require(status == hal::StatusId::degraded && setup.resident_prepared &&
                     setup.resident.plan.prepared_program.stage_count == 1 &&
                     setup.resident.plan.prepared_program.stages[0].semantic ==
                         hal::SemanticEffectId::mirror &&
                     setup.readiness.selected_backend == hal::BackendId::vulkan,
                 "missing optional model disabled an unrelated Vulkan mirror"))
      return false;
    bool skipped = false;
    for (std::size_t index = 0;
         index < setup.resident.plan.resolution.stage_count; ++index) {
      const auto &stage = setup.resident.plan.resolution.stages[index];
      if (stage.intent.semantic == semantic)
        skipped = stage.skipped_optional;
    }
    if (!Require(skipped,
                 "missing optional model was replaced or silently promoted"))
      return false;
    std::vector<std::uint8_t> input(64 * 48 * 3, 29), output(32 * 24 * 3);
    const auto frame = manager.Execute(Binding(request, input, &output, 1));
    if (!Require(frame.status == hal::StatusId::executed &&
                     frame.output_ready && frame.stage_count == 1 &&
                     frame.stages[0].semantic ==
                         hal::SemanticEffectId::mirror &&
                     frame.transfers.uploads == 1 &&
                     frame.transfers.final_readbacks == 1 &&
                     frame.transfers.intermediate_readbacks == 0 &&
                     frame.transfers.cpu_tail_boundaries == 0 &&
                     manager.CommitOutput(frame.completion, true, &reason) ==
                         hal::StatusId::ready &&
                     manager.Snapshot()
                             .resident.latest_committed_frame.stage_count == 1,
                 "optional setup isolation did not execute/attribute the "
                 "surviving effect"))
      return false;
    const auto shutdown = manager.Shutdown();
    if (!Require(shutdown.quiescent &&
                     shutdown.driver_context_release_permitted &&
                     !shutdown.pending,
                 "optional isolation cleanup remained unsafe"))
      return false;
  }
  auto effects = Base(options);
  effects.video_noise_removal.enabled = true;
  effects.video_noise_removal.model_id = kMissing;
  Case missing{"all_optional_missing", effects};
  auto request = Request(missing);
  video::ProductionVideoBackendManager manager(
      vulkan::CreateProductionVideoBackendSetupSource());
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto status = manager.Setup(request, &evidence, &reason);
  const auto setup = manager.Snapshot();
  if (!Require(status == hal::StatusId::degraded &&
                   setup.resident.plan.route ==
                       video::VideoExecutionRoute::passthrough &&
                   setup.readiness.selected_backend == hal::BackendId::invalid,
               "all-optional missing model did not retain neutral passthrough"))
    return false;
  std::vector<std::uint8_t> input(64 * 48 * 3, 17), output(32 * 24 * 3);
  const auto frame = manager.Execute(Binding(request, input, &output, 1));
  if (!Require(
          frame.status == hal::StatusId::passthrough && frame.passthrough &&
              !frame.output_ready && frame.transfers.uploads == 0 &&
              frame.transfers.final_readbacks == 0 &&
              manager.CommitOutput(frame.completion, true, &reason) ==
                  hal::StatusId::ready &&
              manager.Snapshot().resident.latest_committed_frame.stage_count ==
                  0,
          "neutral passthrough invented GPU output or active stage "
          "attribution"))
    return false;
  (void)manager.Shutdown();
  request.resident.plan.intent.stages[0].intent.required = true;
  request.resident.plan.intent.stages[0].intent.failure_action =
      hal::FailureAction::fail_closed;
  video::ProductionVideoBackendManager required(
      vulkan::CreateProductionVideoBackendSetupSource());
  const auto required_status = required.Setup(request, &evidence, &reason);
  if (!Require(required_status != hal::StatusId::ready &&
                   required_status != hal::StatusId::degraded &&
                   !required.Snapshot().resident_prepared,
               "required missing model was silently downgraded to optional"))
    return false;
  (void)required.Shutdown();
  return true;
}

bool BudgetRejection(const Options &options) {
  auto effects = Base(options);
  effects.video_noise_removal.enabled = effects.eye_contact.enabled =
      effects.auto_frame.enabled = true;
  effects.virtual_background.mode = Background::blur;
  effects.virtual_key_light.enabled = effects.vignette.enabled =
      effects.mirror = true;
  for (unsigned axis = 0; axis < 6; ++axis) {
    auto request = Request(Case{"budget_rejection", effects});
    auto &budget = request.resident.plan.budget;
    if (axis == 0)
      budget.persistent_bytes = 1;
    else if (axis == 1)
      budget.scratch_bytes = 1;
    else if (axis == 2)
      budget.upload_staging_bytes = 1;
    else if (axis == 3)
      budget.readback_staging_bytes = 1;
    else if (axis == 4)
      budget.resource_count = 1;
    else
      budget.in_flight_slots = 0;
    video::ProductionVideoBackendManager manager(
        vulkan::CreateProductionVideoBackendSetupSource());
    video::ProductionVideoBackendSetupEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status = manager.Setup(request, &evidence, &reason);
    std::cout << "budget_axis=" << axis
              << " status=" << static_cast<unsigned>(status)
              << " reason=" << static_cast<unsigned>(reason) << '\n';
    if (!Require(status != hal::StatusId::ready &&
                     status != hal::StatusId::degraded &&
                     !manager.Snapshot().resident_prepared,
                 "native preparation enlarged or ignored the caller's "
                 "deficient resource reservation"))
      return false;
    const auto shutdown = manager.Shutdown();
    if (!Require(!shutdown.pending,
                 "rejected budget left native cleanup pending"))
      return false;
  }
  return true;
}

bool UnsafeTimeoutRetirement(const Options &options, bool full = false) {
  auto effects = Base(options);
  // Keep this fault-injection test smaller than one native wait. Injection
  // skips the original frame's wait, whereas a real timeout already spent it.
  effects.video_noise_removal.enabled = true;
  if (full) {
    effects.eye_contact.enabled = effects.auto_frame.enabled = true;
    effects.virtual_background.mode = Background::blur;
    effects.virtual_key_light.enabled = effects.vignette.enabled =
        effects.mirror = true;
    effects.vignette.center_on_tracked_face = true;
  }
  Case test{"unsafe_timeout", effects, 64, 64, 64, 64};
  auto source = std::make_unique<ObservedSource>();
  auto *observed = source.get();
  video::ProductionVideoBackendManager manager(std::move(source));
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(manager.Setup(Request(test), nullptr, &reason) ==
                   hal::StatusId::ready,
               "timeout regression requires the actual prepared native owner"))
    return false;
  auto *factory = dynamic_cast<vulkan::VulkanVideoExecutionFactory *>(
      observed->lease.factory.factory);
  vulkan::VulkanProductionModelDiagnostics before, after;
  std::string error;
  if (!Require(factory && Collect(*observed, &before) &&
                   vulkan::VulkanProductionTesting::InjectNextSubmissionTimeout(
                       *factory, &error),
               "could not arm the existing native fence-timeout test seam"))
    return false;
  std::vector<std::uint8_t> input(64u * 64u * 3u, 0);
  std::vector<std::uint8_t> output(input.size(), 0x5a);
  const auto request = Request(test);
  const auto failed = manager.Execute(Binding(request, input, &output, 1));
  if (!Require(failed.status == hal::StatusId::quarantined &&
                   failed.reason == hal::ReasonId::synchronization_failed &&
                   !failed.output_ready &&
                   !manager.Snapshot().resident.latest_committed_frame.valid &&
                   manager.Snapshot().resident.telemetry.active_backend_frames == 0 &&
                   std::all_of(output.begin(), output.end(),
                               [](auto byte) { return byte == 0x5a; }) &&
                   Collect(*observed, &after) &&
                   !after.device.context_healthy &&
                   after.completed_batches == before.completed_batches,
               "a submitted timeout published output/activity or lost its poison"))
    return false;
  const auto same_counts = [](const auto &a, const auto &b) {
    return a.inference_count == b.inference_count &&
           a.completion_count == b.completion_count &&
           a.warmup_inference_count == b.warmup_inference_count;
  };
  const auto no_success = [&]() {
    return same_counts(before.matting, after.matting) &&
           same_counts(before.denoise, after.denoise) &&
           same_counts(before.face, after.face) &&
           same_counts(before.left_eye, after.left_eye) &&
           same_counts(before.right_eye, after.right_eye);
  };
  if (!Require(no_success(), "timed-out queued models promoted inference success"))
    return false;
  const auto retry = manager.Execute(Binding(request, input, &output, 2));
  if (!Require(retry.status != hal::StatusId::executed &&
                   retry.status != hal::StatusId::degraded &&
                   retry.reason == hal::ReasonId::synchronization_failed &&
                   !retry.output_ready && Collect(*observed, &after) &&
                   no_success(),
               "a poisoned context accepted another frame"))
    return false;
  const auto started = Now(nullptr);
  auto shutdown = manager.Shutdown();
  std::cout << "unsafe_timeout full=" << full
            << " queued=1 output=0 activity=0 success_promotion=0"
               " retry_rejected=1 shutdown_attempt=1 shutdown_ns="
            << Now(nullptr) - started << " quiescent=" << shutdown.quiescent
            << " release=" << shutdown.driver_context_release_permitted
            << " pending=" << shutdown.pending << '\n';
  if (full && shutdown.pending) {
    // A second explicitly requested shutdown is a separate bounded attempt;
    // the poisoned execution path stays closed throughout this late proof.
    const auto retry_started = Now(nullptr);
    shutdown = manager.Shutdown();
    std::cout << "unsafe_timeout full=1 shutdown_attempt=2 shutdown_ns="
              << Now(nullptr) - retry_started
              << " quiescent=" << shutdown.quiescent
              << " release=" << shutdown.driver_context_release_permitted
              << " pending=" << shutdown.pending << '\n';
  }
  return Require(shutdown.quiescent &&
                     shutdown.driver_context_release_permitted &&
                     !shutdown.pending,
                 "bounded late retirement could not safely destroy the failed batch");
}
} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "FAIL invalid or incomplete smoke arguments\n";
    return 2;
  }
  if (!options.require) {
    std::cout << "[SKIP] explicit --require and an isolated preinstalled model "
                 "root are required\n";
    return 77;
  }
  const auto root = options.model_root.lexically_normal();
  if (!Require(root.is_absolute() && root.filename() == "open_video" &&
                   root.parent_path().filename() == "models" &&
                   root.parent_path().parent_path().filename() ==
                       "studiocast" &&
                   std::filesystem::is_directory(root) &&
                   !options.replacement.empty(),
               "absolute model root must end in studiocast/models/open_video; "
               "replacement image is required"))
    return 2;
  if (!Require(!options.case_name.empty() || !options.face_path.empty(),
               "the complete matrix requires a hash-verified local face "
               "fixture; use --case for bounded non-face evidence"))
    return 2;
  if (setenv("XDG_DATA_HOME",
             root.parent_path().parent_path().parent_path().c_str(), 1) != 0)
    return 2;
  video::Rgb24Frame face;
  if (!LoadFace(options, &face))
    return 1;
  const auto cases = Matrix(options);
  unsigned executed = 0;
  std::vector<std::uint8_t> positive, control;
  for (const auto &test : cases) {
    if (!options.case_name.empty() && options.case_name != test.name)
      continue;
    auto *output = test.name == "eye_face"        ? &positive
                   : test.name == "eye_face_zero" ? &control
                                                  : nullptr;
    if (!RunCase(options, test, face, output))
      return 1;
    ++executed;
  }
  if (!options.prepare_only && !positive.empty()) {
    if (control.empty()) {
      const auto zero =
          std::find_if(cases.begin(), cases.end(), [](const auto &test) {
            return test.name == "eye_face_zero";
          });
      if (zero == cases.end() || !RunCase(options, *zero, face, &control))
        return 1;
      ++executed;
    }
    std::size_t changed = 0;
    for (std::size_t index = 0;
         index < positive.size() && index < control.size(); ++index)
      changed += positive[index] != control[index];
    std::cout << "face_positive_final_rgb_changed_bytes=" << changed
              << " detector_readback=none\n";
    if (!Require(positive.size() == control.size() && changed > 0,
                 "face-positive Eye100 produced no final RGB correction "
                 "relative to exact Eye0 control"))
      return 1;
  }
  if (options.case_name.empty() || options.case_name == "isolation") {
    if (!options.prepare_only && !SetupIsolation(options))
      return 1;
    ++executed;
  }
  if (options.case_name.empty() || options.case_name == "budgets") {
    if (!BudgetRejection(options))
      return 1;
    ++executed;
  }
  if (options.case_name.empty() || options.case_name == "unsafe_timeout") {
    if (!options.prepare_only && !UnsafeTimeoutRetirement(options))
      return 1;
    ++executed;
  }
  if (options.case_name.empty() || options.case_name == "unsafe_timeout_all") {
    if (!options.prepare_only && !UnsafeTimeoutRetirement(options, true))
      return 1;
    ++executed;
  }
  if (!Require(executed > 0, "unknown or unavailable selected case"))
    return 2;
  std::cout << "Vulkan effects production matrix passed cases=" << executed
            << " writer=in_memory_not_v4l2 prepare_only="
            << options.prepare_only
            << " camera_quality_vendor_qualification=not_claimed\n";
  return 0;
}
