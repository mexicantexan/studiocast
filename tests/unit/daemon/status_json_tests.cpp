#include "core/util/json.h"
#include "daemon/status_json.h"

#include <array>
#include <cstdlib>
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

using studiocast::hal::BackendId;
using studiocast::hal::ReasonId;
using studiocast::hal::SemanticEffectId;
using studiocast::hal::SemanticEffectProviderId;
using studiocast::hal::StageExecutionStatus;
using studiocast::hal::StatusId;
using studiocast::util::json::Value;
using studiocast::video::CapabilityRefreshSnapshot;
using studiocast::video::CapabilityRefreshStatus;
using studiocast::video::ProductCapabilityDisposition;
using studiocast::video::ProductCapabilityGenerations;
using studiocast::video::ProductCapabilityLane;
using studiocast::video::ProductCapabilityLaneId;
using studiocast::video::ProductCapabilitySnapshot;
using studiocast::video::VideoRuntimeLifecycle;
using studiocast::video::VideoRuntimeStatusSnapshot;
using Code = studiocast::video::effects::EffectCapabilityCode;

bool Expect(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

ProductCapabilityGenerations Generations(std::uint64_t capability = 7) {
  ProductCapabilityGenerations value;
  value.capability = capability;
  value.configuration = 11;
  value.provider = 12;
  value.device = 13;
  value.context = 14;
  return value;
}

CapabilityRefreshSnapshot CurrentRefresh(std::uint64_t generation = 7) {
  CapabilityRefreshSnapshot value;
  value.status = CapabilityRefreshStatus::succeeded;
  value.code = Code::capability_snapshot_current;
  value.requested_generation = generation;
  value.completed_generation = generation;
  return value;
}

ProductCapabilitySnapshot CurrentCapabilities() {
  return studiocast::video::MakeFailClosedProductCapabilitySnapshot(
      Generations(), CurrentRefresh());
}

ProductCapabilityLane *Lane(ProductCapabilitySnapshot *snapshot,
                            SemanticEffectId effect,
                            ProductCapabilityLaneId lane) {
  return studiocast::video::FindProductCapabilityLane(snapshot, effect, lane);
}

void Promote(ProductCapabilitySnapshot *snapshot, SemanticEffectId effect,
             ProductCapabilityLaneId lane, SemanticEffectProviderId provider,
             Code code, StatusId status = StatusId::ready) {
  ProductCapabilityLane *record = Lane(snapshot, effect, lane);
  if (!record)
    std::abort();
  record->disposition =
      status == StatusId::degraded
          ? ProductCapabilityDisposition::usable_with_degraded_behavior
          : ProductCapabilityDisposition::production_usable;
  record->code = code;
  record->nested_code = Code::none;
  record->semantic.semantic = effect;
  record->semantic.provider = provider;
  record->semantic.status = status;
  record->semantic.reason = ReasonId::none;
  record->semantic.live_stage = true;
  record->semantic.production_eligible = true;
  record->semantic.exact_evidence = true;
  if (lane != ProductCapabilityLaneId::open_vulkan)
    return;

  constexpr std::array common{
      studiocast::hal::CapabilityEvidenceCode::vulkan_build_enabled,
      studiocast::hal::CapabilityEvidenceCode::vulkan_loader_available,
      studiocast::hal::CapabilityEvidenceCode::vulkan_physical_device_available,
      studiocast::hal::CapabilityEvidenceCode::vulkan_non_cpu_device,
      studiocast::hal::CapabilityEvidenceCode::vulkan_compute_queue_available,
      studiocast::hal::CapabilityEvidenceCode::vulkan_context_healthy,
      studiocast::hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready,
      studiocast::hal::CapabilityEvidenceCode::vulkan_exact_device_shared,
      studiocast::hal::CapabilityEvidenceCode::vulkan_device_local_residency,
      studiocast::hal::CapabilityEvidenceCode::vulkan_bounded_allocations,
      studiocast::hal::CapabilityEvidenceCode::vulkan_no_cpu_layers,
      studiocast::hal::CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
  };
  for (const auto evidence_code : common) {
    record->semantic.evidence[record->semantic.evidence_count++] = {
        evidence_code, StatusId::ready,   ReasonId::none,
        effect,        BackendId::vulkan, true};
  }
  const auto exact =
      effect == SemanticEffectId::mirror
          ? studiocast::hal::CapabilityEvidenceCode::
                open_vulkan_mirror_production_ready
          : studiocast::hal::CapabilityEvidenceCode::
                open_vulkan_vignette_fixed_center_production_ready;
  record->semantic.evidence[record->semantic.evidence_count++] = {
      exact, StatusId::ready, ReasonId::none, effect, BackendId::vulkan, true};
}

studiocast::hal::ProgramInvalidationKey
Invalidation(const ProductCapabilityGenerations &generations) {
  studiocast::hal::ProgramInvalidationKey value;
  value.generations.configuration = generations.configuration;
  value.generations.geometry = 21;
  value.generations.model = 22;
  value.generations.provider = generations.provider;
  value.generations.device = generations.device;
  value.generations.context = generations.context;
  value.canonical_plan_hash = 31;
  value.model_artifact_hash = 32;
  value.execution_domain_hash = 33;
  return value;
}

VideoRuntimeStatusSnapshot BaseSnapshot() {
  VideoRuntimeStatusSnapshot value;
  value.status_generation = 99;
  value.generations = Generations();
  value.lifecycle = VideoRuntimeLifecycle::idle;
  value.capabilities = CurrentCapabilities();
  value.resolution.status = StatusId::passthrough;
  value.resolution.passthrough = true;
  return value;
}

void AddResolvedStage(VideoRuntimeStatusSnapshot *snapshot,
                      std::uint32_t stage_id, SemanticEffectId effect,
                      BackendId backend, SemanticEffectProviderId provider) {
  auto &stage = snapshot->resolution.stages[snapshot->resolution.stage_count++];
  stage.intent.stage_id = stage_id;
  stage.intent.semantic = effect;
  stage.status = StatusId::ready;
  stage.selected_backend = backend;
  stage.provider = provider;
}

void ConfigureReadyProgram(VideoRuntimeStatusSnapshot *snapshot,
                           BackendId selected_backend) {
  snapshot->resolution.status = StatusId::ready;
  snapshot->resolution.available = true;
  snapshot->resolution.passthrough = false;
  snapshot->resolution.selected_backend = selected_backend;
  snapshot->readiness.configured = true;
  snapshot->readiness.program_id = 44;
  snapshot->readiness.invalidation_key = Invalidation(snapshot->generations);
  snapshot->readiness.value.status = StatusId::ready;
  snapshot->readiness.value.available = true;
  snapshot->readiness.value.selected_backend = selected_backend;
}

void AddCommittedStage(
    VideoRuntimeStatusSnapshot *snapshot, std::uint32_t stage_id,
    SemanticEffectId effect, BackendId backend,
    SemanticEffectProviderId provider,
    StageExecutionStatus status = StageExecutionStatus::succeeded) {
  auto &frame = snapshot->latest_committed_frame;
  frame.valid = true;
  frame.program_id = snapshot->readiness.program_id;
  frame.invalidation_key = snapshot->readiness.invalidation_key;
  frame.execution_domain = snapshot->readiness.value.execution_domain;
  frame.capture_sequence = 101;
  frame.output_commit_sequence = 8;
  frame.stages[frame.stage_count++] = {
      stage_id,
      effect,
      studiocast::hal::PrimitiveOperationId::invalid,
      backend,
      provider,
      status,
      status == StageExecutionStatus::succeeded
          ? ReasonId::none
          : ReasonId::optional_stage_failed_open,
  };
}

VideoRuntimeStatusSnapshot ReadySnapshot() {
  auto value = BaseSnapshot();
  Promote(&value.capabilities, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan,
          SemanticEffectProviderId::open_vulkan_production_kernels,
          Code::open_vulkan_mirror_production_ready);
  AddResolvedStage(&value, 1, SemanticEffectId::mirror, BackendId::vulkan,
                   SemanticEffectProviderId::open_vulkan_production_kernels);
  ConfigureReadyProgram(&value, BackendId::vulkan);
  return value;
}

VideoRuntimeStatusSnapshot MixedSnapshot() {
  auto value = ReadySnapshot();
  value.lifecycle = VideoRuntimeLifecycle::running;
  Promote(&value.capabilities, SemanticEffectId::eye_contact,
          ProductCapabilityLaneId::maxine, SemanticEffectProviderId::maxine_ar,
          Code::maxine_effect_usable);
  AddResolvedStage(&value, 2, SemanticEffectId::eye_contact, BackendId::maxine,
                   SemanticEffectProviderId::maxine_ar);
  value.resolution.selected_backend = BackendId::invalid;
  value.readiness.value.selected_backend = BackendId::invalid;
  AddCommittedStage(&value, 1, SemanticEffectId::mirror, BackendId::vulkan,
                    SemanticEffectProviderId::open_vulkan_production_kernels);
  AddCommittedStage(&value, 2, SemanticEffectId::eye_contact, BackendId::maxine,
                    SemanticEffectProviderId::maxine_ar);
  return value;
}

VideoRuntimeStatusSnapshot BlockedSnapshot() {
  auto value = BaseSnapshot();
  value.lifecycle = VideoRuntimeLifecycle::failed;
  value.resolution.status = StatusId::blocked;
  value.resolution.reason = ReasonId::semantic_provider_unavailable;
  value.resolution.passthrough = false;
  value.resolution.request_mode =
      studiocast::hal::BackendRequestMode::explicit_backend;
  value.resolution.requested_backend = BackendId::vulkan;
  AddResolvedStage(&value, 1, SemanticEffectId::virtual_background_blur,
                   BackendId::invalid, SemanticEffectProviderId::invalid);
  value.resolution.stages[0].status = StatusId::blocked;
  value.resolution.stages[0].reason = ReasonId::semantic_provider_unavailable;
  return value;
}

VideoRuntimeStatusSnapshot StaleSnapshot() {
  auto value = ReadySnapshot();
  value.readiness = {};
  value.lifecycle = VideoRuntimeLifecycle::idle;
  value.capabilities.refresh.status = CapabilityRefreshStatus::stale;
  value.capabilities.refresh.code = Code::capability_snapshot_stale;
  value.capabilities.refresh.requested_generation = 6;
  value.capabilities.refresh.completed_generation = 6;
  return value;
}

VideoRuntimeStatusSnapshot ActiveSnapshot() {
  auto value = BaseSnapshot();
  value.lifecycle = VideoRuntimeLifecycle::running;
  Promote(&value.capabilities, SemanticEffectId::virtual_background_blur,
          ProductCapabilityLaneId::open_cuda,
          SemanticEffectProviderId::open_video_onnxruntime_cuda,
          Code::open_cuda_effect_available);
  AddResolvedStage(&value, 1, SemanticEffectId::virtual_background_blur,
                   BackendId::open_video_onnxruntime,
                   SemanticEffectProviderId::open_video_onnxruntime_cuda);
  ConfigureReadyProgram(&value, BackendId::open_video_onnxruntime);
  AddCommittedStage(&value, 1, SemanticEffectId::virtual_background_blur,
                    BackendId::open_video_onnxruntime,
                    SemanticEffectProviderId::open_video_onnxruntime_cuda);
  return value;
}

VideoRuntimeStatusSnapshot InactiveTelemetrySnapshot() {
  auto value = ActiveSnapshot();
  value.latest_committed_frame = {};
  value.telemetry.selected_backend = BackendId::open_video_onnxruntime;
  value.telemetry.active_backend = BackendId::open_video_onnxruntime;
  value.telemetry.active_provider =
      SemanticEffectProviderId::open_video_onnxruntime_cuda;
  value.telemetry.active_backend_frames = 30;
  value.telemetry.committed_output_frames = 30;
  value.telemetry.transfers.uploads = 30;
  value.telemetry.transfers.final_readbacks = 30;
  return value;
}

studiocast::hal::ExecutionDomainId PreparedVulkanDomain() {
  studiocast::hal::ExecutionDomainId domain;
  domain.backend = BackendId::vulkan;
  domain.api_family = studiocast::hal::ApiFamily::vulkan;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cb1;
  domain.physical_device.stable_id[0] = 0xca;
  domain.logical_context_generation = 17;
  domain.allocator_family = studiocast::hal::AllocatorFamily::vulkan;
  domain.memory_handle_family =
      studiocast::hal::MemoryHandleFamily::vulkan_buffer;
  domain.queue_or_stream_identity = 19;
  domain.ownership_model = studiocast::hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      studiocast::hal::SynchronizationProtocol::vulkan_queue_fence;
  domain.adapter_instance_id = 23;
  return domain;
}

void SetSingleLatencySample(studiocast::hal::LatencyHistogram *histogram,
                            std::uint64_t elapsed_ns) {
  if (!histogram)
    std::abort();
  histogram->samples = 1;
  histogram->total_ns = elapsed_ns;
  histogram->maximum_ns = elapsed_ns;
  histogram->p95_ns = elapsed_ns;
  histogram->p99_ns = elapsed_ns;
  histogram->bucket_upper_ns[0] = elapsed_ns;
  histogram->bucket_counts[0] = 1;
}

studiocast::video::PreparedHardwareExecutionSnapshot
LivePreparedHardwareSnapshot() {
  using namespace studiocast;
  video::PreparedHardwareExecutionSnapshot snapshot;
  snapshot.route = video::PreparedHardwareRoute::hal_live;
  snapshot.eligibility = video::PreparedHardwareEligibility::eligible;
  snapshot.prepared = true;
  snapshot.setup.mode = video::HardwareExecutionSetupMode::hal_live;
  snapshot.setup.status = hal::StatusId::ready;
  snapshot.setup.reason = hal::ReasonId::none;
  snapshot.setup.shadow_compared = false;

  const auto domain = PreparedVulkanDomain();
  auto &resolution = snapshot.setup.resolution;
  resolution.request_mode = hal::BackendRequestMode::explicit_backend;
  resolution.strict_backend_selection = true;
  resolution.status = hal::StatusId::ready;
  resolution.requested_backend = hal::BackendId::vulkan;
  resolution.selected_backend = hal::BackendId::vulkan;
  resolution.selected_provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  resolution.available = true;
  auto &resolved = resolution.stages[resolution.stage_count++];
  resolved.intent.stage_id = 9;
  resolved.intent.semantic = hal::SemanticEffectId::mirror;
  resolved.status = hal::StatusId::ready;
  resolved.selected_backend = hal::BackendId::vulkan;
  resolved.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  resolved.adapter_instance_id = domain.adapter_instance_id;
  resolved.execution_domain = domain;

  auto &plan = snapshot.setup.domain_plan;
  plan.status = hal::StatusId::ready;
  plan.expected_transfers.uploads = 1;
  plan.expected_transfers.final_readbacks = 1;
  plan.expected_transfers.synchronizations = 1;
  auto &island = plan.islands[plan.island_count++];
  island.adapter_instance_id = domain.adapter_instance_id;
  island.execution_domain = domain;
  island.resolved_stage_indices[island.stage_count++] = 0;
  island.requires_upload = true;
  island.requires_final_readback = true;

  snapshot.readiness.status = hal::StatusId::ready;
  snapshot.readiness.requested_backend = hal::BackendId::vulkan;
  snapshot.readiness.selected_backend = hal::BackendId::vulkan;
  snapshot.readiness.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  snapshot.readiness.execution_domain = domain;
  snapshot.readiness.ready_stage_mask = 1;
  snapshot.readiness.available = true;

  snapshot.active_program.valid = true;
  snapshot.active_program.program_id = 29;
  snapshot.active_program.owner_adapter_instance_id =
      domain.adapter_instance_id;
  snapshot.active_program.execution_domain = domain;
  snapshot.active_program.stage_count = 1;
  snapshot.active_program.invalidation_key.generations.configuration = 31;
  snapshot.active_program.invalidation_key.generations.geometry = 32;
  snapshot.active_program.invalidation_key.generations.provider = 33;
  snapshot.active_program.invalidation_key.generations.device = 34;
  snapshot.active_program.invalidation_key.generations.context =
      domain.logical_context_generation;
  snapshot.active_program.invalidation_key.canonical_plan_hash = 35;
  snapshot.active_program.invalidation_key.execution_domain_hash = 36;

  auto &telemetry = snapshot.telemetry;
  telemetry.requested_backend = hal::BackendId::vulkan;
  telemetry.selected_backend = hal::BackendId::vulkan;
  telemetry.active_backend = hal::BackendId::vulkan;
  telemetry.active_provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  telemetry.execution_attempts = 1;
  telemetry.execution_successes = 1;
  telemetry.committed_output_frames = 1;
  telemetry.active_backend_frames = 1;
  telemetry.setup_attempts = 1;
  telemetry.setup_successes = 1;
  telemetry.runtime_status = hal::StatusId::executed;
  telemetry.transfers.uploads = 1;
  telemetry.transfers.final_readbacks = 1;
  telemetry.transfers.synchronizations = 1;
  SetSingleLatencySample(&telemetry.frame_latency, 100'000);
  SetSingleLatencySample(&telemetry.stage_latency[0], 80'000);
  telemetry.committed_stage_successes[0] = 1;

  auto &commit = snapshot.latest_committed_frame;
  commit.valid = true;
  commit.program_id = 29;
  commit.execution_domain = domain;
  commit.invalidation_key = snapshot.active_program.invalidation_key;
  commit.capture_sequence = 37;
  commit.output_commit_sequence = 1;
  commit.stages[commit.stage_count++] = {
      9,
      hal::SemanticEffectId::mirror,
      hal::PrimitiveOperationId::invalid,
      hal::BackendId::vulkan,
      hal::SemanticEffectProviderId::open_vulkan_production_kernels,
      hal::StageExecutionStatus::succeeded,
      hal::ReasonId::none,
  };
  return snapshot;
}

bool ParseJson(const std::string &text, Value *value,
               std::string_view context) {
  std::string error;
  const bool ok = studiocast::util::json::Parse(text, value, &error);
  if (!ok)
    std::cerr << "FAIL: " << context << ": " << error << '\n';
  return ok;
}

bool IsSubset(const Value &expected, const Value &actual,
              const std::string &path) {
  if (const auto *object = expected.AsObject()) {
    const auto *actual_object = actual.AsObject();
    if (!actual_object) {
      std::cerr << "FAIL: " << path << " is not an object\n";
      return false;
    }
    bool ok = true;
    for (const auto &[key, expected_value] : *object) {
      const auto found = actual_object->find(key);
      if (found == actual_object->end()) {
        std::cerr << "FAIL: missing " << path << '.' << key << '\n';
        ok = false;
        continue;
      }
      ok &= IsSubset(expected_value, found->second, path + '.' + key);
    }
    return ok;
  }
  if (const auto *array = expected.AsArray()) {
    const auto *actual_array = actual.AsArray();
    if (!actual_array || actual_array->size() != array->size()) {
      std::cerr << "FAIL: array mismatch at " << path << '\n';
      return false;
    }
    bool ok = true;
    for (std::size_t i = 0; i < array->size(); ++i)
      ok &= IsSubset((*array)[i], (*actual_array)[i],
                     path + '[' + std::to_string(i) + ']');
    return ok;
  }
  bool equal = false;
  if (expected.IsNull())
    equal = actual.IsNull();
  else if (const auto *expected_bool = expected.AsBool())
    equal = actual.AsBool() && *actual.AsBool() == *expected_bool;
  else if (const auto *expected_number = expected.AsNumber())
    equal = actual.AsNumber() && *actual.AsNumber() == *expected_number;
  else if (const auto *expected_string = expected.AsString())
    equal = actual.AsString() && *actual.AsString() == *expected_string;
  if (!equal)
    std::cerr << "FAIL: scalar mismatch at " << path << '\n';
  return equal;
}

bool MatchesFixture(std::string_view fixture,
                    const VideoRuntimeStatusSnapshot &snapshot) {
  const auto result =
      studiocast::daemon::SerializeVideoRuntimeStatusJson(snapshot);
  if (!Expect(result.ok, "valid snapshot must serialize"))
    return false;
  Value actual;
  if (!ParseJson(result.json, &actual, "serialized runtime status"))
    return false;
  const std::filesystem::path path =
      std::filesystem::path(STUDIOCAST_SOURCE_DIR) / "tests" / "data" /
      "status" / fixture;
  std::ifstream input(path);
  if (!input) {
    std::cerr << "FAIL: cannot read fixture " << path << '\n';
    return false;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  Value expected;
  if (!ParseJson(buffer.str(), &expected, path.string()))
    return false;
  return IsSubset(expected, actual, "$");
}

bool TestGoldenCompatibilityScenarios() {
  bool ok = true;
  ok &= MatchesFixture("ready.json", ReadySnapshot());
  ok &= MatchesFixture("mixed.json", MixedSnapshot());
  ok &= MatchesFixture("blocked.json", BlockedSnapshot());
  ok &= MatchesFixture("stale.json", StaleSnapshot());
  ok &= MatchesFixture("active.json", ActiveSnapshot());
  ok &= MatchesFixture("inactive_with_telemetry.json",
                       InactiveTelemetrySnapshot());
  return ok;
}

bool TestCompleteTypedCapabilityProjection() {
  auto snapshot = CurrentCapabilities();
  Promote(&snapshot, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan,
          SemanticEffectProviderId::open_vulkan_production_kernels,
          Code::open_vulkan_mirror_production_ready);
  const auto result =
      studiocast::daemon::SerializeProductCapabilityStatusJson(snapshot);
  bool ok = Expect(result.ok, "complete capability matrix must serialize");
  Value root;
  ok &= ParseJson(result.json, &root, "serialized capability status");
  const auto *object = root.AsObject();
  const Value *lanes_value = nullptr;
  if (object) {
    const auto found = object->find("lanes");
    if (found != object->end())
      lanes_value = &found->second;
  }
  const auto *lanes = lanes_value ? lanes_value->AsArray() : nullptr;
  ok &= Expect(lanes && lanes->size() ==
                            studiocast::video::kProductCapabilityRecordCount,
               "typed projector must preserve the exact 9x4 matrix");
  ok &= Expect(
      result.json.find("\"token\":\"open_vulkan_mirror_production_ready\"") !=
          std::string::npos,
      "stable Vulkan success token must be projected");
  ok &= Expect(result.json.find("\"code\":\"vulkan_exact_device_shared\"") !=
                   std::string::npos,
               "typed Vulkan evidence must be projected");
  ok &= Expect(result.json.find("effect.virtual_background.blur.cpu."
                                "no_selectable_production_path") !=
                   std::string::npos,
               "CPU product-effect blocker must remain explicit");
  return ok;
}

bool TestMalformedSnapshotsFailClosed() {
  bool ok = true;
  auto capabilities = CurrentCapabilities();
  capabilities.lane_count--;
  auto capability_result =
      studiocast::daemon::SerializeProductCapabilityStatusJson(capabilities);
  ok &= Expect(!capability_result.ok && capability_result.json.empty(),
               "missing capability lane must not produce JSON");

  auto runtime = ActiveSnapshot();
  runtime.status_generation = 0;
  auto runtime_result =
      studiocast::daemon::SerializeVideoRuntimeStatusJson(runtime);
  ok &= Expect(!runtime_result.ok && runtime_result.json.empty(),
               "invalid status generation must not produce JSON");

  runtime = ActiveSnapshot();
  runtime.capabilities.lanes[1] = runtime.capabilities.lanes[0];
  runtime_result = studiocast::daemon::SerializeVideoRuntimeStatusJson(runtime);
  ok &= Expect(!runtime_result.ok && runtime_result.json.empty(),
               "duplicate typed capability lane must fail closed");
  return ok;
}

bool TestActivityRequiresGenerationValidSuccessfulCommit() {
  bool ok = true;
  auto runtime = ActiveSnapshot();
  runtime.latest_committed_frame.invalidation_key.generations.context++;
  auto result = studiocast::daemon::SerializeVideoRuntimeStatusJson(runtime);
  ok &= Expect(result.ok, "stale commit must not invalidate status snapshot");
  ok &= Expect(result.json.find("\"active_backend\":\"\"") != std::string::npos,
               "wrong-generation commit must not claim active backend");
  ok &=
      Expect(result.json.find("\"effects_backends\":\"\"") != std::string::npos,
             "wrong-generation commit must not claim active effects");

  runtime = ActiveSnapshot();
  runtime.latest_committed_frame.stages[0].status =
      StageExecutionStatus::optional_failed_open;
  runtime.latest_committed_frame.stages[0].reason =
      ReasonId::optional_stage_failed_open;
  result = studiocast::daemon::SerializeVideoRuntimeStatusJson(runtime);
  ok &= Expect(result.ok && result.json.find("\"effects_backends\":\"\"") !=
                                std::string::npos,
               "failed-open stage must not receive active attribution");
  return ok;
}

bool TestPreparedHardwareLiveCommittedSuccess() {
  const auto result =
      studiocast::daemon::SerializePreparedHardwareExecutionJson(
          LivePreparedHardwareSnapshot(), true, false);
  bool ok = true;
  ok &= Expect(result.ok && result.snapshot_valid &&
                   result.active_attribution_valid,
               "live committed prepared snapshot must expose attribution");
  Value parsed;
  ok &= ParseJson(result.json, &parsed, "prepared hardware status");
  ok &= Expect(result.json.find("\"token\":\"hal_live\"") != std::string::npos,
               "live route token must be serialized");
  ok &= Expect(
      result.json.find("\"active_backend\":{\"id\":5,\"token\":\"vulkan\"}") !=
          std::string::npos,
      "successful committed effect must attribute Vulkan");
  ok &= Expect(
      result.json.find("\"uploads\":1") != std::string::npos &&
          result.json.find("\"final_readbacks\":1") != std::string::npos &&
          result.json.find("\"synchronizations\":1") != std::string::npos,
      "exact cumulative transfer counters must be serialized");
  ok &= Expect(result.json.find("\"p95_ns\":100000") != std::string::npos &&
                   result.json.find("\"p99_ns\":100000") != std::string::npos &&
                   result.json.find("\"stage_timing\"") != std::string::npos,
               "tail latency and stage timing must be serialized");
  ok &= Expect(
      result.json.find("\"shadow_compared\":false") != std::string::npos &&
          result.json.find("\"shadow_comparison\"") == std::string::npos,
      "unperformed shadow comparison must not claim a result");
  ok &= Expect(result.json.size() < 64u * 1024u,
               "prepared hardware status payload must remain bounded");
  return ok;
}

bool TestPreparedHardwareActivityFailsClosed() {
  bool ok = true;
  auto snapshot = LivePreparedHardwareSnapshot();
  auto result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, false, false);
  ok &= Expect(
      result.ok && !result.active_attribution_valid &&
          result.json.find("\"active_backend\":{\"id\":0,\"token\":\"\"}") !=
              std::string::npos &&
          result.json.find("\"current_program_match\":true") !=
              std::string::npos,
      "stopped-after-commit snapshot must not remain active");

  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, true);
  ok &= Expect(!result.active_attribution_valid,
               "starting supervisor must not expose active attribution");

  snapshot = {};
  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(
      result.ok && result.snapshot_valid && !result.active_attribution_valid &&
          result.json.find("\"token\":\"legacy_shadow\"") !=
              std::string::npos &&
          result.json.find("\"token\":\"unavailable\"") != std::string::npos,
      "legacy unprepared route must retain fail-closed route diagnostics");

  snapshot = LivePreparedHardwareSnapshot();
  snapshot.latest_committed_frame = {};
  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(!result.active_attribution_valid,
               "uncommitted execution must not become active");

  snapshot = LivePreparedHardwareSnapshot();
  ++snapshot.latest_committed_frame.invalidation_key.generations.configuration;
  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(!result.snapshot_valid && !result.active_attribution_valid &&
                   result.validation_code == "stale_committed_output",
               "stale invalidation generation must fail closed");

  snapshot = LivePreparedHardwareSnapshot();
  ++snapshot.latest_committed_frame.execution_domain.queue_or_stream_identity;
  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(!result.snapshot_valid && !result.active_attribution_valid &&
                   result.validation_code == "wrong_execution_domain",
               "wrong committed execution domain must fail closed");

  snapshot = LivePreparedHardwareSnapshot();
  snapshot.telemetry.runtime_status = StatusId::quarantined;
  snapshot.telemetry.runtime_reason = ReasonId::device_lost;
  snapshot.telemetry.device_losses = 1;
  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(!result.active_attribution_valid &&
                   result.json.find("\"token\":\"device_lost\"") !=
                       std::string::npos,
               "device loss quarantine must clear active attribution");
  return ok;
}

bool TestPreparedHardwareOptionalFailOpenAndMalformedCounts() {
  bool ok = true;
  auto snapshot = LivePreparedHardwareSnapshot();
  auto &commit_stage = snapshot.latest_committed_frame.stages[0];
  commit_stage.status = StageExecutionStatus::optional_failed_open;
  commit_stage.reason = ReasonId::optional_stage_failed_open;
  snapshot.telemetry.active_backend = BackendId::invalid;
  snapshot.telemetry.active_provider = SemanticEffectProviderId::invalid;
  snapshot.telemetry.active_backend_frames = 0;
  snapshot.telemetry.optional_fail_open_stages = 1;
  snapshot.telemetry.committed_stage_successes[0] = 0;
  snapshot.telemetry.runtime_status = StatusId::degraded;
  snapshot.telemetry.runtime_reason = ReasonId::optional_stage_failed_open;
  auto result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(
      result.ok && result.snapshot_valid && !result.active_attribution_valid &&
          result.json.find("\"committed_stages\":[]") != std::string::npos,
      "fail-open preserved output may remain committed but failed "
      "stage must not receive active attribution");
  ok &= Expect(result.json.find("\"optional_fail_open_stages\":1") !=
                   std::string::npos,
               "fail-open evidence must remain explicit");

  snapshot = LivePreparedHardwareSnapshot();
  snapshot.latest_committed_frame.stage_count =
      snapshot.latest_committed_frame.stages.size() + 1;
  result = studiocast::daemon::SerializePreparedHardwareExecutionJson(
      snapshot, true, false);
  ok &= Expect(result.ok && !result.snapshot_valid &&
                   !result.active_attribution_valid &&
                   result.validation_code == "malformed_count",
               "malformed counts must serialize bounded diagnostics and fail "
               "closed");
  Value parsed;
  ok &= ParseJson(result.json, &parsed, "malformed prepared hardware status");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestGoldenCompatibilityScenarios();
  ok &= TestCompleteTypedCapabilityProjection();
  ok &= TestMalformedSnapshotsFailClosed();
  ok &= TestActivityRequiresGenerationValidSuccessfulCommit();
  ok &= TestPreparedHardwareLiveCommittedSuccess();
  ok &= TestPreparedHardwareActivityFailsClosed();
  ok &= TestPreparedHardwareOptionalFailOpenAndMalformedCounts();
  if (ok)
    std::cout << "All status JSON projector tests passed.\n";
  return ok ? 0 : 1;
}
