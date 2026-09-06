#include "core/video/effects/broadcast_effects.h"
#include "core/video/production_video_backend_manager.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace studiocast::video {
std::unique_ptr<IProductionVideoBackendSetupSource>
CreateStaticProductionVideoBackendSetupSource() noexcept;
}

namespace {
namespace hal = studiocast::hal;
namespace video = studiocast::video;

struct Options {
  std::string model_id = "modnet-webnn-256-fp32";
  std::filesystem::path model_root;
  std::string effects = "blur";
  bool automatic = false;
  std::uint32_t input_width = 64;
  std::uint32_t input_height = 48;
  std::uint32_t output_width = 64;
  std::uint32_t output_height = 48;
  std::uint32_t frames = 3;
  bool require = false;
};

bool ParseUnsigned(std::string_view value, std::uint32_t *out) {
  const auto result = std::from_chars(value.data(), value.data() + value.size(), *out);
  return result.ec == std::errc{} && result.ptr == value.data() + value.size();
}

bool ParseOptions(int argc, char **argv, Options *options, std::string *error) {
  options->require = std::getenv("STUDIOCAST_REQUIRE_OPEN_CUDA_PRODUCTION_SMOKE") != nullptr &&
                     std::string_view(std::getenv("STUDIOCAST_REQUIRE_OPEN_CUDA_PRODUCTION_SMOKE")) == "1";
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    const auto take = [&](std::string_view name, std::string *value) {
      if (arg != name || i + 1 >= argc)
        return false;
      *value = argv[++i];
      return true;
    };
    std::string value;
    if (take("--model-id", &options->model_id)) {
    } else if (take("--model-root", &value)) {
      options->model_root = value;
    } else if (take("--effects", &options->effects)) {
    } else if (take("--input", &value) || take("--output", &value)) {
      const bool input = arg == "--input";
      const auto separator = value.find('x');
      if (separator == std::string::npos ||
          !ParseUnsigned(std::string_view(value).substr(0, separator),
                         input ? &options->input_width : &options->output_width) ||
          !ParseUnsigned(std::string_view(value).substr(separator + 1),
                         input ? &options->input_height : &options->output_height)) {
        *error = std::string(arg) + " must be WIDTHxHEIGHT";
        return false;
      }
    } else if (take("--frames", &value)) {
      if (!ParseUnsigned(value, &options->frames)) {
        *error = "--frames must be an integer";
        return false;
      }
    } else if (arg == "--auto") {
      options->automatic = true;
    } else if (arg == "--cuda") {
      options->automatic = false;
    } else if (arg == "--require") {
      options->require = true;
    } else {
      *error = "unknown or incomplete argument: " + std::string(arg);
      return false;
    }
  }
  if (options->effects != "blur" && options->effects != "keylight" &&
      options->effects != "blur+keylight") {
    *error = "--effects must be blur, keylight, or blur+keylight";
    return false;
  }
  if (options->input_width == 0 || options->input_height == 0 ||
      options->output_width == 0 || options->output_height == 0 ||
      options->input_width > 4096 || options->input_height > 4096 ||
      options->output_width > 4096 || options->output_height > 4096 ||
      options->frames == 0 || options->frames > 8) {
    *error = "dimensions must be in [1,4096] and --frames must be in [1,8]";
    return false;
  }
  return true;
}

hal::ImageDescriptor HostRgb(std::uint32_t width, std::uint32_t height,
                             hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = width;
  image.height = height;
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
  image.planes[0].row_stride_bytes = static_cast<std::uint64_t>(width) * 3;
  image.planes[0].size_bytes = image.planes[0].row_stride_bytes * height;
  image.planes[0].width = width;
  image.planes[0].height = height;
  image.planes[0].alignment_bytes = 1;
  return image;
}

void AddSigned(video::CanonicalVideoStage *stage,
               video::CanonicalVideoParameterId id, std::int64_t value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = video::CanonicalVideoParameterKind::signed_integer;
  parameter.signed_value = value;
}

void AddUnsigned(video::CanonicalVideoStage *stage,
                 video::CanonicalVideoParameterId id, std::uint64_t value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = video::CanonicalVideoParameterKind::unsigned_integer;
  parameter.unsigned_value = value;
}

void AddBoolean(video::CanonicalVideoStage *stage,
                video::CanonicalVideoParameterId id, bool value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = video::CanonicalVideoParameterKind::boolean;
  parameter.boolean_value = value;
}

bool StopNever(void *) noexcept { return false; }
int wait_context = 0;
std::uint64_t Now(void *) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
void Poll(void *, std::uint64_t wait_nanoseconds) noexcept {
  std::this_thread::sleep_for(std::chrono::nanoseconds(wait_nanoseconds));
}

video::ProductionVideoResidentSetup BuildRequest(const Options &options) {
  video::ProductionVideoResidentSetup request;
  request.wait.timeout_nanoseconds = UINT64_C(5000000000);
  request.wait.poll_interval_nanoseconds = UINT64_C(1000000);
  request.wait.maximum_poll_count = 5000;
  request.wait.stop_requested = &StopNever;
  request.wait.cancel_requested = &StopNever;
  request.wait.now_nanoseconds = &Now;
  request.wait.poll_wait = &Poll;
  request.wait.callback_context = &wait_context;
  request.wait.callback_context_generation = 1;
  request.wait.callback_context_lifetime = hal::ResourceLifetime::adapter_context;
  auto &plan = request.plan;
  plan.plan_id = 1;
  plan.intent.backend.mode = options.automatic
                                 ? hal::BackendRequestMode::automatic
                                 : hal::BackendRequestMode::explicit_backend;
  plan.intent.backend.backend =
      options.automatic ? hal::BackendId::invalid : hal::BackendId::cuda;
  plan.intent.backend.strict = !options.automatic;
  plan.intent.preferences.compute_family =
      options.automatic ? hal::ComputeFamilyAllowance::any
                        : hal::ComputeFamilyAllowance::cuda_family;
  plan.intent.preferences.effects_engine =
      options.automatic ? hal::EffectsEnginePreferenceConstraint::automatic
                        : hal::EffectsEnginePreferenceConstraint::open_cuda;
  plan.intent.preferences.strict_effects_engine = !options.automatic;
  plan.intent.requested_provider =
      options.automatic ? hal::SemanticEffectProviderId::invalid
                        : hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  plan.intent.strict_provider_selection = !options.automatic;

  request.effects.engine = options.automatic
                               ? video::effects::EffectsEnginePreference::auto_select
                               : video::effects::EffectsEnginePreference::open_cuda;
  request.effects.virtual_background.model_id = options.model_id;
  const bool blur = options.effects != "keylight";
  const bool keylight = options.effects != "blur";
  if (blur) {
    request.effects.virtual_background.mode =
        video::effects::VirtualBackgroundMode::blur;
    auto &stage = plan.intent.stages[plan.intent.stage_count++];
    stage.intent.stage_id = 1;
    stage.intent.semantic = hal::SemanticEffectId::virtual_background_blur;
    stage.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
    AddSigned(&stage, video::CanonicalVideoParameterId::strength,
              request.effects.virtual_background.strength);
    AddUnsigned(&stage, video::CanonicalVideoParameterId::remove_color_rgb, 0);
    AddUnsigned(&stage, video::CanonicalVideoParameterId::greenscreen_mode,
                static_cast<std::uint64_t>(
                    request.effects.virtual_background.greenscreen_mode));
    AddBoolean(&stage, video::CanonicalVideoParameterId::greenscreen_temporal,
               request.effects.virtual_background.greenscreen_temporal);
  }
  if (keylight) {
    request.effects.virtual_key_light.enabled = true;
    auto &stage = plan.intent.stages[plan.intent.stage_count++];
    stage.intent.stage_id = static_cast<std::uint32_t>(plan.intent.stage_count);
    stage.intent.semantic = hal::SemanticEffectId::virtual_key_light;
    stage.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
    AddSigned(&stage, video::CanonicalVideoParameterId::intensity,
              request.effects.virtual_key_light.intensity);
    AddSigned(&stage, video::CanonicalVideoParameterId::temperature_preset,
              request.effects.virtual_key_light.temperature_preset);
    AddSigned(&stage, video::CanonicalVideoParameterId::direction_pan_degrees,
              request.effects.virtual_key_light.direction_pan_degrees);
  }
  plan.host_input = HostRgb(options.input_width, options.input_height,
                            hal::Mutability::read_only);
  plan.host_output = HostRgb(options.output_width, options.output_height,
                             hal::Mutability::write_only);
  plan.invalidation_key.generations.configuration = 1;
  plan.invalidation_key.generations.geometry = 1;
  plan.invalidation_key.generations.provider = 1;
  plan.invalidation_key.generations.device = 1;
  plan.invalidation_key.generations.context = 1;
  plan.invalidation_key.canonical_plan_hash = 1;
  constexpr std::uint64_t kGiB = UINT64_C(1024) * 1024u * 1024u;
  plan.budget.persistent_bytes = 8u * kGiB;
  plan.budget.scratch_bytes = 4u * kGiB;
  plan.budget.upload_staging_bytes = kGiB;
  plan.budget.readback_staging_bytes = kGiB;
  plan.budget.resource_count = 1024;
  plan.budget.in_flight_slots = 1;
  plan.cpu_tails[0].id = hal::CpuTailId::backend_resolved_output;
  plan.cpu_tails[0].after_stage_id =
      plan.intent.stages[plan.intent.stage_count - 1].intent.stage_id;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].boundary_index =
      static_cast<std::uint32_t>(plan.intent.stage_count);
  plan.cpu_tail_count = 1;
  plan.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  plan.migration_policy = video::VideoExecutionMigrationPolicy::prepared_only;
  return request;
}

std::string Hex(const std::array<std::uint8_t, 32> &bytes) {
  std::ostringstream stream;
  for (const auto byte : bytes)
    stream << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<unsigned>(byte);
  return stream.str();
}

int Run(const Options &options) {
  if (!options.model_root.empty()) {
    const auto normalized = options.model_root.lexically_normal();
    if (normalized.filename() != "open_video" ||
        normalized.parent_path().filename() != "models" ||
        normalized.parent_path().parent_path().filename() != "studiocast") {
      std::cerr << "FAIL model_root must end in studiocast/models/open_video\n";
      return 2;
    }
    const auto xdg = normalized.parent_path().parent_path().parent_path();
    if (setenv("XDG_DATA_HOME", xdg.c_str(), 1) != 0) {
      std::cerr << "FAIL could not set XDG_DATA_HOME for read-only model root\n";
      return 2;
    }
  }

  auto resident = BuildRequest(options);
  video::ProductionVideoBackendSetupRequest request;
  request.kind = video::ProductionVideoBackendSetupKind::resident_island;
  request.resident = resident;
  video::ProductionVideoBackendManager manager(
      video::CreateStaticProductionVideoBackendSetupSource());
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto started = std::chrono::steady_clock::now();
  const auto status = manager.Setup(request, &evidence, &reason);
  const auto setup_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started)
                            .count();
  const auto setup = manager.Snapshot();
  std::cout << "setup status=" << static_cast<unsigned>(status)
            << " reason=" << static_cast<unsigned>(reason)
            << " elapsed_ns=" << setup_ns
            << " requested=" << static_cast<unsigned>(setup.readiness.requested_backend)
            << " selected=" << static_cast<unsigned>(setup.readiness.selected_backend)
            << " provider=" << static_cast<unsigned>(setup.readiness.provider)
            << " domain_backend="
            << static_cast<unsigned>(setup.readiness.execution_domain.backend)
            << " model_id=" << options.model_id << " input="
            << options.input_width << 'x' << options.input_height << " output="
            << options.output_width << 'x' << options.output_height
            << " prepared=" << setup.resident_prepared
            << " source_called=" << evidence.setup_source_called
            << " candidate_setup_only=" << evidence.candidate_setup_only
            << " candidate_rejected=" << evidence.candidate_rejected
            << " candidate_status="
            << static_cast<unsigned>(evidence.candidate_readiness.status)
            << " candidate_reason="
            << static_cast<unsigned>(evidence.candidate_readiness.reason)
            << " candidate_selected="
            << static_cast<unsigned>(evidence.candidate_readiness.selected_backend)
            << " candidate_provider="
            << static_cast<unsigned>(evidence.candidate_readiness.provider)
            << '\n';
  for (std::size_t i = 0; i < setup.resident.plan.intent.stage_count; ++i) {
    const auto &stage = setup.resident.plan.intent.stages[i];
    std::cout << "stage index=" << i
              << " semantic=" << static_cast<unsigned>(stage.intent.semantic)
              << " artifact_generation=" << stage.model.artifact_generation
              << " artifact_set_sha256=" << Hex(stage.model.artifact_set_hash)
              << '\n';
  }
  if (status != hal::StatusId::ready || !setup.resident_prepared) {
    std::cerr << "FAIL production setup did not produce a prepared resident program"
              << " status=" << static_cast<unsigned>(status)
              << " reason=" << static_cast<unsigned>(reason) << '\n';
    return 1;
  }
  if (setup.readiness.selected_backend != hal::BackendId::cuda ||
      setup.readiness.provider !=
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda) {
    std::cerr << "FAIL prepared route did not select CUDA and its CUDA provider\n";
    return 1;
  }
  const auto &domain = setup.readiness.execution_domain;
  std::cout << "domain api_family=" << static_cast<unsigned>(domain.api_family)
            << " vendor_id=" << domain.physical_device.vendor_id
            << " device_id=" << domain.physical_device.device_id
            << " context_generation=" << domain.logical_context_generation
            << " allocator=" << static_cast<unsigned>(domain.allocator_family)
            << " memory_handle=" << static_cast<unsigned>(domain.memory_handle_family)
            << " queue_or_stream=" << domain.queue_or_stream_identity
            << " ownership=" << static_cast<unsigned>(domain.ownership_model)
            << " synchronization="
            << static_cast<unsigned>(domain.synchronization_protocol)
            << " adapter=" << domain.adapter_instance_id << '\n';

  const auto input_bytes = static_cast<std::size_t>(options.input_width) *
                           options.input_height * 3;
  const auto output_bytes = static_cast<std::size_t>(options.output_width) *
                            options.output_height * 3;
  std::vector<std::byte> input(input_bytes);
  std::vector<std::byte> output(output_bytes);
  for (std::uint32_t frame_index = 0; frame_index < options.frames; ++frame_index) {
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<std::byte>((i + frame_index * 17u) & 0xffu);
    hal::HostFrameBinding binding;
    binding.input_descriptor = resident.plan.host_input;
    binding.output_descriptor = resident.plan.host_output;
    binding.input_planes[0].read_data = input.data();
    binding.input_planes[0].span_bytes = input.size();
    binding.input_planes[0].row_stride_bytes = options.input_width * 3u;
    binding.output_planes[0].write_data = output.data();
    binding.output_planes[0].span_bytes = output.size();
    binding.output_planes[0].row_stride_bytes = options.output_width * 3u;
    binding.input_plane_count = binding.output_plane_count = 1;
    binding.ownership = hal::OwnershipModel::borrowed_until_completion;
    binding.lifetime = hal::ResourceLifetime::frame_borrowed;
    binding.capture_sequence = frame_index + 1;
    const auto frame = manager.Execute(binding);
    std::cout << "frame sequence=" << binding.capture_sequence
              << " status=" << static_cast<unsigned>(frame.status)
              << " reason=" << static_cast<unsigned>(frame.reason)
              << " output_ready=" << frame.output_ready
              << " elapsed_ns=" << frame.frame_elapsed_ns
              << " uploads=" << frame.transfers.uploads
              << " final_readbacks=" << frame.transfers.final_readbacks
              << " intermediate_readbacks=" << frame.transfers.intermediate_readbacks
              << " intermediate_uploads=" << frame.transfers.intermediate_uploads
              << " synchronizations=" << frame.transfers.synchronizations << '\n';
    for (std::size_t stage_index = 0; stage_index < frame.stage_count;
         ++stage_index) {
      const auto &stage = frame.stages[stage_index];
      std::cout << "frame_stage index=" << stage_index
                << " semantic=" << static_cast<unsigned>(stage.semantic)
                << " status=" << static_cast<unsigned>(stage.status)
                << " reason=" << static_cast<unsigned>(stage.reason)
                << " elapsed_ns=" << stage.elapsed_ns << '\n';
    }
    if (frame.status != hal::StatusId::executed || !frame.output_ready) {
      std::cerr << "FAIL frame execution did not produce writable RGB output\n";
      return 1;
    }
    const auto before = manager.Snapshot().resident.latest_committed_frame;
    reason = hal::ReasonId::none;
    const auto commit = manager.CommitOutput(frame.completion, true, &reason);
    const auto after = manager.Snapshot().resident.latest_committed_frame;
    std::cout << "commit transport=in_memory status="
              << static_cast<unsigned>(commit)
              << " reason=" << static_cast<unsigned>(reason)
              << " before_valid=" << before.valid << " after_valid=" << after.valid
              << " committed_sequence=" << after.capture_sequence << '\n';
    if (commit != hal::StatusId::ready || !after.valid ||
        after.capture_sequence != binding.capture_sequence) {
      std::cerr << "FAIL in-memory writer commit did not activate the frame\n";
      return 1;
    }
  }
  const auto final = manager.Snapshot().resident.telemetry;
  std::uint64_t checksum = UINT64_C(14695981039346656037);
  for (const auto byte : output) {
    checksum ^= static_cast<std::uint8_t>(byte);
    checksum *= UINT64_C(1099511628211);
  }
  std::cout << "summary writer=in_memory_not_v4l2 attempts="
            << final.execution_attempts << " successes=" << final.execution_successes
            << " committed=" << final.committed_output_frames
            << " active_backend_frames=" << final.active_backend_frames
            << " fallback_frames=" << final.fallback_frames
            << " uploads=" << final.transfers.uploads
            << " final_readbacks=" << final.transfers.final_readbacks
            << " synchronizations=" << final.transfers.synchronizations
            << " cpu_tail_boundaries=" << final.transfers.cpu_tail_boundaries
            << " latency_samples=" << final.frame_latency.samples
            << " latency_total_ns=" << final.frame_latency.total_ns
            << " output_rgb_fnv1a64=" << std::hex << checksum << std::dec << '\n';
  return final.committed_output_frames == options.frames &&
                 final.active_backend_frames == options.frames &&
                 final.fallback_frames == 0
             ? 0
             : 1;
}
} // namespace

int main(int argc, char **argv) {
  Options options;
  std::string error;
  if (!ParseOptions(argc, argv, &options, &error)) {
    std::cerr << "FAIL " << error << '\n';
    return 2;
  }
  if (!options.require) {
    std::cout << "[SKIP] Open CUDA production smoke requires "
                 "STUDIOCAST_REQUIRE_OPEN_CUDA_PRODUCTION_SMOKE=1 or --require\n";
    return 77;
  }
  return Run(options);
}
