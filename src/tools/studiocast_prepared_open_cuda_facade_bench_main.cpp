#include "core/video/prepared_open_cuda_hardware_execution_owner.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

namespace hal = studiocast::hal;
namespace open_video = studiocast::open_video;
namespace video = studiocast::video;

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kMaximumDimension = 4096;
constexpr std::uint64_t kMaximumWarmup = 1'000'000;
constexpr std::uint64_t kMaximumIterations = 1'000'000;
constexpr std::uint64_t kAdapterInstanceId = 0xFADE71;

std::atomic<std::uint64_t> g_allocation_count{0};
std::atomic<bool> g_count_allocations{false};

void *Allocate(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_acquire))
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void *AllocateAligned(std::size_t size, std::size_t alignment) {
  if (g_count_allocations.load(std::memory_order_acquire))
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  void *memory = nullptr;
  if (posix_memalign(&memory, alignment, std::max(size, std::size_t{1})) == 0)
    return memory;
  throw std::bad_alloc();
}

struct Options {
  std::uint32_t width = 1280;
  std::uint32_t height = 720;
  std::uint64_t warmup = 30;
  std::uint64_t iterations = 300;
  hal::SemanticEffectId semantic =
      hal::SemanticEffectId::virtual_background_blur;
  bool csv = false;
};

struct Statistics {
  double average_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double minimum_ms = 0.0;
  double maximum_ms = 0.0;
  double frames_per_second = 0.0;
};

struct CounterTotals {
  std::uint64_t uploads = 0;
  std::uint64_t final_readbacks = 0;
  std::uint64_t intermediate_readbacks = 0;
  std::uint64_t intermediate_uploads = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t writer_copy_tails = 0;
};

enum class RunFailure : std::uint8_t {
  none = 0,
  setup,
  frame,
  transfer_contract,
  output_content,
  commit,
  allocation,
  adapter_call_count,
  telemetry,
  attribution,
};

bool ParseUnsigned(std::string_view text, std::uint64_t *value) {
  if (!value || text.empty())
    return false;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

void PrintUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " [--warmup N] [--iterations N] [--width W] [--height H]"
               " [--effect blur|remove] [--csv]\n";
}

bool ParseOptions(int argc, char **argv, Options *options) {
  if (!options)
    return false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--csv") {
      options->csv = true;
      continue;
    }
    if (argument == "--help" || argument == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--effect") {
      if (++index >= argc)
        return false;
      const std::string_view effect(argv[index]);
      if (effect == "blur") {
        options->semantic = hal::SemanticEffectId::virtual_background_blur;
      } else if (effect == "remove") {
        options->semantic = hal::SemanticEffectId::virtual_background_remove;
      } else {
        return false;
      }
      continue;
    }
    if (argument != "--warmup" && argument != "--iterations" &&
        argument != "--width" && argument != "--height") {
      return false;
    }
    if (++index >= argc)
      return false;
    std::uint64_t value = 0;
    if (!ParseUnsigned(argv[index], &value))
      return false;
    if (argument == "--warmup") {
      if (value > kMaximumWarmup)
        return false;
      options->warmup = value;
    } else if (argument == "--iterations") {
      if (value == 0 || value > kMaximumIterations)
        return false;
      options->iterations = value;
    } else {
      if (value == 0 || value > kMaximumDimension)
        return false;
      const auto dimension = static_cast<std::uint32_t>(value);
      if (argument == "--width")
        options->width = dimension;
      else
        options->height = dimension;
    }
  }
  return true;
}

bool FrameBytes(const Options &options, std::size_t *row_bytes,
                std::size_t *frame_bytes) {
  if (!row_bytes || !frame_bytes ||
      options.width >
          std::numeric_limits<std::size_t>::max() / std::size_t{3}) {
    return false;
  }
  *row_bytes = static_cast<std::size_t>(options.width) * 3;
  if (options.height > std::numeric_limits<std::size_t>::max() / *row_bytes) {
    return false;
  }
  *frame_bytes = *row_bytes * static_cast<std::size_t>(options.height);
  return *frame_bytes != 0;
}

const char *EffectName(hal::SemanticEffectId semantic) {
  return semantic == hal::SemanticEffectId::virtual_background_remove
             ? "virtual_background_remove"
             : "virtual_background_blur";
}

const char *FailureName(RunFailure failure) {
  switch (failure) {
  case RunFailure::none:
    return "none";
  case RunFailure::setup:
    return "setup";
  case RunFailure::frame:
    return "frame";
  case RunFailure::transfer_contract:
    return "transfer_contract";
  case RunFailure::output_content:
    return "output_content";
  case RunFailure::commit:
    return "commit";
  case RunFailure::allocation:
    return "allocation";
  case RunFailure::adapter_call_count:
    return "adapter_call_count";
  case RunFailure::telemetry:
    return "telemetry";
  case RunFailure::attribution:
    return "attribution";
  }
  return "unknown";
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = 0xFA;
  domain.physical_device.stable_id[1] = 0xCE;
  domain.logical_context_generation = 9;
  domain.allocator_family = hal::AllocatorFamily::cuda_driver;
  domain.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = 0xFADECA71;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = kAdapterInstanceId;
  return domain;
}

open_video::ValidatedModelArtifactIdentity Artifact() {
  open_video::ValidatedModelArtifactIdentity artifact;
  artifact.registry_generation = 7;
  artifact.model_id_sha256[0] = 0x11;
  artifact.manifest_sha256[0] = 0x22;
  artifact.artifact_sha256[0] = 0x33;
  artifact.artifact_set_sha256[0] = 0x44;
  return artifact;
}

open_video::ModelPack
Pack(const open_video::ValidatedModelArtifactIdentity &artifact) {
  open_video::ModelPack pack;
  pack.id = "synthetic-open-cuda-matting";
  pack.display_name = "Synthetic Open CUDA matting";
  pack.task = "matting";
  pack.registry_generation = artifact.registry_generation;
  pack.registry_manifest_sha256 = artifact.manifest_sha256;
  open_video::ModelFile file;
  file.name = "synthetic-model.onnx";
  file.kind = "onnx";
  file.role = "main";
  pack.files.push_back(file);
  open_video::MattingSpec matting;
  matting.input.name = "input";
  matting.input.layout = "nchw";
  matting.input.dtype = "float32";
  matting.input.width = 16;
  matting.input.height = 8;
  matting.input.channels = 3;
  matting.output.name = "alpha";
  matting.output.kind = "alpha";
  matting.output.dtype = "float32";
  matting.output.layout = "nchw";
  matting.output.width = 16;
  matting.output.height = 8;
  matting.output.channels = 1;
  matting.preprocess.color = "rgb";
  matting.preprocess.range = "0..1";
  pack.matting = matting;
  return pack;
}

bool NeverRequested(void *) noexcept { return false; }

std::uint64_t SyntheticNow(void *) noexcept {
  static std::atomic<std::uint64_t> now{0};
  return now.fetch_add(1000, std::memory_order_relaxed);
}

void SyntheticPoll(void *, std::uint64_t) noexcept {}

video::PreparedOpenCudaHardwareRequest Request(const Options &options,
                                               std::size_t frame_bytes) {
  const auto artifact = Artifact();
  video::PreparedOpenCudaHardwareRequest request;
  request.effects_engine =
      video::PreparedOpenCudaEffectsEngineIntent::open_cuda;
  request.compute = video::PreparedOpenCudaComputeIntent::cuda;
  request.requested_backend = hal::BackendId::cuda;
  request.strict_backend_selection = true;
  request.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  request.strict_provider_selection = true;
  request.stage_id = 3;
  request.semantic = options.semantic;
  request.semantic_stage_count = 1;
  request.single_resident_island = true;
  request.capture_width = options.width;
  request.capture_height = options.height;
  request.output_width = options.width;
  request.output_height = options.height;
  request.host_input_is_tight_rgb24 = true;
  request.host_output_is_tight_rgb24 = true;
  request.generations = {11, 12, 13, 14, 15, 0};
  request.capacity.persistent_bytes = static_cast<std::uint64_t>(frame_bytes);
  request.capacity.scratch_bytes = 4096;
  request.capacity.upload_staging_bytes =
      static_cast<std::uint64_t>(frame_bytes);
  request.capacity.readback_staging_bytes =
      static_cast<std::uint64_t>(frame_bytes);
  request.capacity.resource_count = 8;
  request.capacity.in_flight_slots = 1;
  request.model_artifact = artifact;
  request.model_pack = Pack(artifact);
  request.strength = 8;
  request.remove_background_rgb = {0x10, 0x20, 0x30};
  request.wait.timeout_nanoseconds = 5'000'000;
  request.wait.poll_interval_nanoseconds = 50'000;
  request.wait.maximum_poll_count = 100;
  request.wait.stop_requested = NeverRequested;
  request.wait.cancel_requested = NeverRequested;
  request.wait.now_nanoseconds = SyntheticNow;
  request.wait.poll_wait = SyntheticPoll;
  return request;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t resource_id,
                             hal::MemoryDomain memory_domain,
                             std::uint64_t size_bytes) {
  hal::ResourceHandle resource;
  resource.resource_id = resource_id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = memory_domain;
  resource.lifetime = hal::ResourceLifetime::frame_borrowed;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = size_bytes;
  resource.alignment_bytes = 1;
  resource.mutable_resource =
      memory_domain == hal::MemoryDomain::readback_staging;
  return resource;
}

class SyntheticCudaAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override {
    return kAdapterInstanceId;
  }

  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::cuda;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    program_ = program;
    candidate->handle_id = 1;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = adapter_instance_id();
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    readiness->status = hal::StatusId::degraded;
    readiness->reason = hal::ReasonId::cpu_tail_required;
    readiness->requested_backend = hal::BackendId::cuda;
    readiness->selected_backend = hal::BackendId::cuda;
    readiness->provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = 1;
    readiness->available = true;
    readiness->degraded = true;
    if (reason)
      *reason = hal::ReasonId::cpu_tail_required;
    return hal::StatusId::degraded;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    ++bind_calls;
    binding_ = binding;
    tokens->input = Resource(program_, 1000 + binding.capture_sequence,
                             hal::MemoryDomain::upload_staging,
                             binding.input_planes[0].span_bytes);
    tokens->output = Resource(program_, 3'000'000 + binding.capture_sequence,
                              hal::MemoryDomain::readback_staging,
                              binding.output_planes[0].span_bytes);
    output_ = tokens->output;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data,
                static_cast<std::size_t>(binding_.input_planes[0].span_bytes));
    hal::ExecuteResult result;
    result.status = hal::StatusId::executed;
    result.completion.token_id = next_completion_token_++;
    result.completion.program_id = program_.program_id;
    result.completion.adapter_instance_id = adapter_instance_id();
    result.completion.capture_sequence = request.capture_sequence;
    result.completion.complete = true;
    result.output = output_;
    result.output_valid = true;
    result.stage_count = 1;
    result.stages[0].stage_id = program_.stages[0].stage_id;
    result.stages[0].semantic = program_.stages[0].semantic;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    result.transfers.cpu_tail_boundaries = 1;
    return result;
  }

  hal::StatusId ResolveHostFrameOutput(
      const hal::PreparedProgramHandle &, const hal::CompletionToken &,
      const hal::ResourceHandle &, hal::HostFrameOutputResolution *resolution,
      hal::ReasonId *reason) noexcept override {
    ++resolve_calls;
    resolution->descriptor = binding_.output_descriptor;
    resolution->planes = binding_.output_planes;
    resolution->plane_count = binding_.output_plane_count;
    resolution->capture_sequence = binding_.capture_sequence;
    resolution->output_ready = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    return hal::StatusId::ready;
  }

  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
    ++destroy_calls;
    *program = {};
  }

  void Shutdown() noexcept override { ++shutdown_calls; }

  std::uint64_t prepare_calls = 0;
  std::uint64_t bind_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t resolve_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;

private:
  std::uint64_t next_completion_token_ = 1;
  hal::PreparedProgram program_{};
  hal::HostFrameBinding binding_{};
  hal::ResourceHandle output_{};
};

hal::AdapterRegistration Registration(SyntheticCudaAdapter *adapter,
                                      hal::SemanticEffectId semantic) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = adapter->adapter_instance_id();
  registration.backend = hal::BackendId::cuda;
  registration.execution_domain = Domain();
  registration.selection_priority = 10;
  registration.resident_adapter = adapter;
  auto &capability =
      registration
          .semantic_capabilities[registration.semantic_capability_count++];
  capability.semantic = semantic;
  capability.provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  capability.status = hal::StatusId::ready;
  capability.reason = hal::ReasonId::none;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  return registration;
}

bool ExactTransfers(const hal::TransferCounters &transfers) {
  return transfers.uploads == 1 && transfers.final_readbacks == 1 &&
         transfers.intermediate_readbacks == 0 &&
         transfers.intermediate_uploads == 0 &&
         transfers.synchronizations == 1 && transfers.cpu_tail_boundaries == 1;
}

void Accumulate(const hal::TransferCounters &transfers, CounterTotals *totals) {
  totals->uploads += transfers.uploads;
  totals->final_readbacks += transfers.final_readbacks;
  totals->intermediate_readbacks += transfers.intermediate_readbacks;
  totals->intermediate_uploads += transfers.intermediate_uploads;
  totals->synchronizations += transfers.synchronizations;
  totals->writer_copy_tails += transfers.cpu_tail_boundaries;
}

bool RunFrame(video::PreparedOpenCudaHardwareExecutionOwner *owner,
              const Options &options, std::vector<std::uint8_t> *input,
              std::vector<std::uint8_t> *output, std::size_t row_bytes,
              std::uint64_t capture_sequence, CounterTotals *totals,
              std::uint64_t *elapsed_nanoseconds, RunFailure *failure) {
  const auto start = Clock::now();
  const auto frame = owner->ExecuteHostFrame(
      input->data(), input->size(), row_bytes, output->data(), output->size(),
      row_bytes, capture_sequence);
  hal::ReasonId commit_reason = hal::ReasonId::none;
  const auto commit_status =
      frame.output_ready && frame.completion_pending
          ? owner->CommitOutput(frame.completion, true, &commit_reason)
          : hal::StatusId::failed;
  const auto finish = Clock::now();
  if (elapsed_nanoseconds) {
    *elapsed_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
            .count());
  }

  if (frame.status != hal::StatusId::executed || !frame.output_ready ||
      !frame.completion_pending || frame.output != output->data() ||
      frame.output_size_bytes != output->size() ||
      frame.output_row_stride_bytes != row_bytes ||
      frame.stage.semantic != options.semantic ||
      frame.stage.status != hal::StageExecutionStatus::succeeded ||
      !frame.stage.valid_output_preserved) {
    *failure = RunFailure::frame;
    return false;
  }
  if (!ExactTransfers(frame.transfers)) {
    *failure = RunFailure::transfer_contract;
    return false;
  }
  if (commit_status != hal::StatusId::ready ||
      commit_reason != hal::ReasonId::none) {
    *failure = RunFailure::commit;
    return false;
  }
  if (std::memcmp(input->data(), output->data(), input->size()) != 0) {
    *failure = RunFailure::output_content;
    return false;
  }
  Accumulate(frame.transfers, totals);
  return true;
}

double Percentile(const std::vector<std::uint64_t> &sorted, double percentile) {
  const double rank =
      percentile * static_cast<double>(sorted.size() - std::size_t{1});
  const auto lower = static_cast<std::size_t>(rank);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = rank - static_cast<double>(lower);
  const double nanoseconds = static_cast<double>(sorted[lower]) +
                             (static_cast<double>(sorted[upper]) -
                              static_cast<double>(sorted[lower])) *
                                 fraction;
  return nanoseconds / 1'000'000.0;
}

Statistics ComputeStatistics(std::vector<std::uint64_t> *samples) {
  Statistics statistics;
  std::sort(samples->begin(), samples->end());
  const long double total =
      std::accumulate(samples->begin(), samples->end(), 0.0L);
  const long double average_ns =
      total / static_cast<long double>(samples->size());
  statistics.average_ms = static_cast<double>(average_ns / 1'000'000.0L);
  statistics.p50_ms = Percentile(*samples, 0.50);
  statistics.p95_ms = Percentile(*samples, 0.95);
  statistics.p99_ms = Percentile(*samples, 0.99);
  statistics.minimum_ms = static_cast<double>(samples->front()) / 1'000'000.0;
  statistics.maximum_ms = static_cast<double>(samples->back()) / 1'000'000.0;
  if (average_ns > 0)
    statistics.frames_per_second =
        static_cast<double>(1'000'000'000.0L / average_ns);
  return statistics;
}

bool TelemetryValid(
    const video::PreparedOpenCudaHardwareExecutionSnapshot &before,
    const video::PreparedOpenCudaHardwareExecutionSnapshot &after,
    const Options &options) {
  const auto count = options.iterations;
  const auto &first = before.telemetry;
  const auto &last = after.telemetry;
  return last.execution_attempts - first.execution_attempts == count &&
         last.execution_successes - first.execution_successes == count &&
         last.committed_output_frames - first.committed_output_frames ==
             count &&
         last.active_backend_frames - first.active_backend_frames == count &&
         last.fallback_frames - first.fallback_frames == 0 &&
         last.transfers.uploads - first.transfers.uploads == count &&
         last.transfers.final_readbacks - first.transfers.final_readbacks ==
             count &&
         last.transfers.intermediate_readbacks -
                 first.transfers.intermediate_readbacks ==
             0 &&
         last.transfers.intermediate_uploads -
                 first.transfers.intermediate_uploads ==
             0 &&
         last.transfers.synchronizations - first.transfers.synchronizations ==
             count &&
         last.transfers.cpu_tail_boundaries -
                 first.transfers.cpu_tail_boundaries ==
             count;
}

bool AttributionValid(
    const video::PreparedOpenCudaHardwareExecutionSnapshot &before,
    const video::PreparedOpenCudaHardwareExecutionSnapshot &after,
    const Options &options) {
  const auto &frame = after.latest_committed_frame;
  return after.telemetry.active_backend == hal::BackendId::cuda &&
         after.telemetry.active_provider ==
             hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
         frame.valid &&
         frame.capture_sequence == options.warmup + options.iterations &&
         frame.output_commit_sequence ==
             before.latest_committed_frame.output_commit_sequence +
                 options.iterations &&
         frame.execution_domain == Domain() && frame.stage_count == 1 &&
         frame.stages[0].semantic == options.semantic &&
         frame.stages[0].backend == hal::BackendId::cuda &&
         frame.stages[0].provider ==
             hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
         frame.stages[0].status == hal::StageExecutionStatus::succeeded;
}

std::uint64_t Checksum(const std::vector<std::uint8_t> &bytes) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (const auto byte : bytes) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

void PrintCsvHeader() {
  std::cout
      << "backend,status,synthetic,effect,width,height,warmup,iterations,"
         "avg_ms,p50_ms,p95_ms,p99_ms,min_ms,max_ms,frames_per_second,"
         "uploads_per_frame,final_readbacks_per_frame,"
         "intermediate_readbacks_per_frame,intermediate_uploads_per_frame,"
         "synchronizations_per_frame,writer_copy_tails_per_frame,"
         "committed_backend,committed_provider,checksum\n";
}

void PrintResult(const Options &options, const Statistics &statistics,
                 std::uint64_t checksum) {
  if (options.csv) {
    PrintCsvHeader();
    std::cout << std::fixed << std::setprecision(9) << "synthetic_fake,ok,true,"
              << EffectName(options.semantic) << ',' << options.width << ','
              << options.height << ',' << options.warmup << ','
              << options.iterations << ',' << statistics.average_ms << ','
              << statistics.p50_ms << ',' << statistics.p95_ms << ','
              << statistics.p99_ms << ',' << statistics.minimum_ms << ','
              << statistics.maximum_ms << ',' << statistics.frames_per_second
              << ",1,1,0,0,1,1,cuda,open_video_onnxruntime_cuda," << checksum
              << '\n';
    return;
  }
  std::cout << "Prepared Open CUDA facade benchmark\n"
            << "  backend/status : synthetic_fake / ok (no CUDA or ORT work)\n"
            << "  contract       : cuda / open_video_onnxruntime_cuda\n"
            << "  effect         : " << EffectName(options.semantic) << '\n'
            << "  geometry       : " << options.width << 'x' << options.height
            << " RGB24\n"
            << "  iterations     : " << options.iterations << " (warmup "
            << options.warmup << ")\n"
            << std::fixed << std::setprecision(6)
            << "  avg/p50        : " << statistics.average_ms << " / "
            << statistics.p50_ms << " ms\n"
            << "  p95/p99        : " << statistics.p95_ms << " / "
            << statistics.p99_ms << " ms\n"
            << "  min/max        : " << statistics.minimum_ms << " / "
            << statistics.maximum_ms << " ms\n"
            << "  frames/second  : " << statistics.frames_per_second << '\n'
            << "  transfers/frame: upload=1 final_readback=1"
               " intermediate_readback=0 intermediate_upload=0\n"
            << "  sync/tails     : synchronization=1 writer_copy_tail=1"
               " per frame\n"
            << "  checksum       : " << checksum << '\n';
}

int Run(const Options &options) {
  std::size_t row_bytes = 0;
  std::size_t frame_bytes = 0;
  if (!FrameBytes(options, &row_bytes, &frame_bytes)) {
    std::cerr << "ERROR: invalid or overflowing frame geometry\n";
    return EXIT_FAILURE;
  }

  std::vector<std::uint8_t> input(frame_bytes);
  std::vector<std::uint8_t> output(frame_bytes, 0);
  std::vector<std::uint64_t> samples(
      static_cast<std::size_t>(options.iterations));
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<std::uint8_t>((index * std::size_t{131} + 17) & 0xff);
  }

  SyntheticCudaAdapter adapter;
  video::PreparedOpenCudaHardwareExecutionOwner owner(
      Registration(&adapter, options.semantic));
  const auto setup = owner.Configure(Request(options, frame_bytes));
  if (setup.route != video::PreparedOpenCudaHardwareRoute::hal_live ||
      setup.status != hal::StatusId::degraded || adapter.prepare_calls != 1) {
    std::cerr << "ERROR: synthetic facade setup failed"
              << " route=" << static_cast<unsigned>(setup.route)
              << " eligibility=" << static_cast<unsigned>(setup.eligibility)
              << " status=" << static_cast<unsigned>(setup.status)
              << " reason=" << static_cast<unsigned>(setup.reason)
              << " prepare_calls=" << adapter.prepare_calls << '\n';
    return EXIT_FAILURE;
  }

  CounterTotals warmup_totals;
  RunFailure failure = RunFailure::none;
  for (std::uint64_t index = 0; index < options.warmup; ++index) {
    if (!RunFrame(&owner, options, &input, &output, row_bytes, index + 1,
                  &warmup_totals, nullptr, &failure)) {
      std::cerr << "ERROR: warmup failed: " << FailureName(failure) << '\n';
      return EXIT_FAILURE;
    }
  }

  const auto before = owner.Snapshot();
  const auto bind_before = adapter.bind_calls;
  const auto execute_before = adapter.execute_calls;
  const auto resolve_before = adapter.resolve_calls;
  const auto prepare_before = adapter.prepare_calls;
  CounterTotals totals;
  failure = RunFailure::none;
  g_allocation_count.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_release);
  for (std::uint64_t index = 0; index < options.iterations; ++index) {
    if (!RunFrame(&owner, options, &input, &output, row_bytes,
                  options.warmup + index + 1, &totals,
                  &samples[static_cast<std::size_t>(index)], &failure)) {
      break;
    }
  }
  g_count_allocations.store(false, std::memory_order_release);

  if (failure == RunFailure::none &&
      g_allocation_count.load(std::memory_order_relaxed) != 0) {
    failure = RunFailure::allocation;
  }
  if (failure == RunFailure::none &&
      (adapter.prepare_calls != prepare_before ||
       adapter.bind_calls - bind_before != options.iterations ||
       adapter.execute_calls - execute_before != options.iterations ||
       adapter.resolve_calls - resolve_before != options.iterations)) {
    failure = RunFailure::adapter_call_count;
  }
  const auto expected = options.iterations;
  if (failure == RunFailure::none &&
      (totals.uploads != expected || totals.final_readbacks != expected ||
       totals.intermediate_readbacks != 0 || totals.intermediate_uploads != 0 ||
       totals.synchronizations != expected ||
       totals.writer_copy_tails != expected)) {
    failure = RunFailure::transfer_contract;
  }

  const auto after = owner.Snapshot();
  if (failure == RunFailure::none && !TelemetryValid(before, after, options)) {
    failure = RunFailure::telemetry;
  }
  if (failure == RunFailure::none &&
      !AttributionValid(before, after, options)) {
    failure = RunFailure::attribution;
  }
  if (failure != RunFailure::none) {
    std::cerr << "ERROR: measured synthetic facade run failed: "
              << FailureName(failure) << " allocations="
              << g_allocation_count.load(std::memory_order_relaxed) << '\n';
    return EXIT_FAILURE;
  }

  const auto statistics = ComputeStatistics(&samples);
  PrintResult(options, statistics, Checksum(output));
  return EXIT_SUCCESS;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void *operator new(std::size_t size, std::align_val_t alignment) {
  return AllocateAligned(size, static_cast<std::size_t>(alignment));
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
  return AllocateAligned(size, static_cast<std::size_t>(alignment));
}
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
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

int main(int argc, char **argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }
  try {
    return Run(options);
  } catch (const std::bad_alloc &) {
    g_count_allocations.store(false, std::memory_order_release);
    std::cerr << "ERROR: benchmark resource allocation failed\n";
    return EXIT_FAILURE;
  }
}
