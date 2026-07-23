#include "daemon/status_json.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string_view>

namespace studiocast::daemon {
namespace {

using video::effects::EffectCapabilityCode;

void AppendJsonString(std::ostream &out, std::string_view value) {
  out << '"';
  for (const char raw : value) {
    const auto c = static_cast<unsigned char>(raw);
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20) {
        const auto flags = out.flags();
        const auto fill = out.fill();
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<unsigned int>(c);
        out.flags(flags);
        out.fill(fill);
      } else {
        out << static_cast<char>(c);
      }
      break;
    }
  }
  out << '"';
}

[[nodiscard]] constexpr std::string_view
BackendToken(hal::BackendId value) noexcept {
  switch (value) {
  case hal::BackendId::invalid:
    return {};
  case hal::BackendId::cpu:
    return "cpu";
  case hal::BackendId::cuda:
    return "cuda";
  case hal::BackendId::maxine:
    return "maxine";
  case hal::BackendId::open_video_onnxruntime:
    return "open_video_onnxruntime";
  case hal::BackendId::vulkan:
    return "vulkan";
  case hal::BackendId::test_fake:
    return "test_fake";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
ProviderToken(hal::SemanticEffectProviderId value) noexcept {
  switch (value) {
  case hal::SemanticEffectProviderId::invalid:
    return {};
  case hal::SemanticEffectProviderId::maxine_vfx:
    return "maxine_vfx";
  case hal::SemanticEffectProviderId::maxine_ar:
    return "maxine_ar";
  case hal::SemanticEffectProviderId::open_video_onnxruntime_cuda:
    return "open_video_onnxruntime_cuda";
  case hal::SemanticEffectProviderId::open_video_onnxruntime_cpu:
    return "open_video_onnxruntime_cpu";
  case hal::SemanticEffectProviderId::open_vulkan_production_kernels:
    return "open_vulkan_production_kernels";
  case hal::SemanticEffectProviderId::maxine_cuda_utility:
    return "maxine_cuda_utility";
  case hal::SemanticEffectProviderId::test_fake:
    return "test_fake";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
StatusToken(hal::StatusId value) noexcept {
  switch (value) {
  case hal::StatusId::invalid:
    return "invalid";
  case hal::StatusId::ready:
    return "ready";
  case hal::StatusId::blocked:
    return "blocked";
  case hal::StatusId::degraded:
    return "degraded";
  case hal::StatusId::unavailable:
    return "unavailable";
  case hal::StatusId::passthrough:
    return "passthrough";
  case hal::StatusId::executed:
    return "executed";
  case hal::StatusId::failed:
    return "failed";
  case hal::StatusId::quarantined:
    return "quarantined";
  case hal::StatusId::stopped:
    return "stopped";
  case hal::StatusId::resource_exhausted:
    return "resource_exhausted";
  case hal::StatusId::stale:
    return "stale";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
ReasonToken(hal::ReasonId value) noexcept {
  switch (value) {
  case hal::ReasonId::none:
    return {};
  case hal::ReasonId::malformed_contract:
    return "malformed_contract";
  case hal::ReasonId::unsupported_contract_version:
    return "unsupported_contract_version";
  case hal::ReasonId::malformed_descriptor:
    return "malformed_descriptor";
  case hal::ReasonId::malformed_execution_domain:
    return "malformed_execution_domain";
  case hal::ReasonId::foreign_adapter:
    return "foreign_adapter";
  case hal::ReasonId::stale_program:
    return "stale_program";
  case hal::ReasonId::wrong_generation:
    return "wrong_generation";
  case hal::ReasonId::wrong_execution_domain:
    return "wrong_execution_domain";
  case hal::ReasonId::resource_exhausted:
    return "resource_exhausted";
  case hal::ReasonId::cross_adapter_residency_forbidden:
    return "cross_adapter_residency_forbidden";
  case hal::ReasonId::required_stage_failed:
    return "required_stage_failed";
  case hal::ReasonId::optional_stage_failed_open:
    return "optional_stage_failed_open";
  case hal::ReasonId::device_lost:
    return "device_lost";
  case hal::ReasonId::temporal_discontinuity:
    return "temporal_discontinuity";
  case hal::ReasonId::reconfiguration_required:
    return "reconfiguration_required";
  case hal::ReasonId::shutdown_requested:
    return "shutdown_requested";
  case hal::ReasonId::output_write_failed:
    return "output_write_failed";
  case hal::ReasonId::output_not_committed:
    return "output_not_committed";
  case hal::ReasonId::incomplete_execution:
    return "incomplete_execution";
  case hal::ReasonId::invalid_resource:
    return "invalid_resource";
  case hal::ReasonId::synchronization_failed:
    return "synchronization_failed";
  case hal::ReasonId::strict_backend_unavailable:
    return "strict_backend_unavailable";
  case hal::ReasonId::cpu_tail_required:
    return "cpu_tail_required";
  case hal::ReasonId::circuit_breaker_open:
    return "circuit_breaker_open";
  case hal::ReasonId::cancellation_unsupported_v1:
    return "cancellation_unsupported_v1";
  case hal::ReasonId::semantic_provider_unavailable:
    return "semantic_provider_unavailable";
  case hal::ReasonId::semantic_effect_unsupported:
    return "semantic_effect_unsupported";
  case hal::ReasonId::capability_evidence_incomplete:
    return "capability_evidence_incomplete";
  case hal::ReasonId::host_frame_binding_unsupported:
    return "host_frame_binding_unsupported";
  case hal::ReasonId::host_frame_binding_invalid:
    return "host_frame_binding_invalid";
  case hal::ReasonId::circuit_breaker_retry_pending:
    return "circuit_breaker_retry_pending";
  case hal::ReasonId::circuit_breaker_permanently_disabled:
    return "circuit_breaker_permanently_disabled";
  case hal::ReasonId::old_program_policy_incompatible:
    return "old_program_policy_incompatible";
  case hal::ReasonId::transfer_contract_violated:
    return "transfer_contract_violated";
  case hal::ReasonId::frame_order_violated:
    return "frame_order_violated";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
EvidenceToken(hal::CapabilityEvidenceCode value) noexcept {
  switch (value) {
  case hal::CapabilityEvidenceCode::invalid:
    return {};
  case hal::CapabilityEvidenceCode::vulkan_build_enabled:
    return "vulkan_build_enabled";
  case hal::CapabilityEvidenceCode::vulkan_loader_available:
    return "vulkan_loader_available";
  case hal::CapabilityEvidenceCode::vulkan_physical_device_available:
    return "vulkan_physical_device_available";
  case hal::CapabilityEvidenceCode::vulkan_non_cpu_device:
    return "vulkan_non_cpu_device";
  case hal::CapabilityEvidenceCode::vulkan_compute_queue_available:
    return "vulkan_compute_queue_available";
  case hal::CapabilityEvidenceCode::vulkan_context_healthy:
    return "vulkan_context_healthy";
  case hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready:
    return "vulkan_utility_kernels_ready";
  case hal::CapabilityEvidenceCode::vulkan_exact_device_shared:
    return "vulkan_exact_device_shared";
  case hal::CapabilityEvidenceCode::vulkan_device_local_residency:
    return "vulkan_device_local_residency";
  case hal::CapabilityEvidenceCode::vulkan_bounded_allocations:
    return "vulkan_bounded_allocations";
  case hal::CapabilityEvidenceCode::vulkan_no_cpu_layers:
    return "vulkan_no_cpu_layers";
  case hal::CapabilityEvidenceCode::vulkan_no_intermediate_readbacks:
    return "vulkan_no_intermediate_readbacks";
  case hal::CapabilityEvidenceCode::open_vulkan_mirror_production_ready:
    return "open_vulkan_mirror_production_ready";
  case hal::CapabilityEvidenceCode::
      open_vulkan_vignette_fixed_center_production_ready:
    return "open_vulkan_vignette_fixed_center_production_ready";
  case hal::CapabilityEvidenceCode::vulkan_matting_runtime_ready:
    return "vulkan_matting_runtime_ready";
  case hal::CapabilityEvidenceCode::vulkan_model_artifacts_valid:
    return "vulkan_model_artifacts_valid";
  case hal::CapabilityEvidenceCode::vulkan_warmup_complete:
    return "vulkan_warmup_complete";
  case hal::CapabilityEvidenceCode::vulkan_live_stage_present:
    return "vulkan_live_stage_present";
  case hal::CapabilityEvidenceCode::vulkan_tracked_vignette_center_supported:
    return "vulkan_tracked_vignette_center_supported";
  case hal::CapabilityEvidenceCode::provider_runtime_available:
    return "provider_runtime_available";
  case hal::CapabilityEvidenceCode::model_artifact_identity_valid:
    return "model_artifact_identity_valid";
  case hal::CapabilityEvidenceCode::synchronization_evidence_valid:
    return "synchronization_evidence_valid";
  case hal::CapabilityEvidenceCode::import_export_interop_valid:
    return "import_export_interop_valid";
  case hal::CapabilityEvidenceCode::vulkan_backend_disabled_in_build:
    return "vulkan_backend_disabled_in_build";
  case hal::CapabilityEvidenceCode::vulkan_runtime_not_found:
    return "vulkan_runtime_not_found";
  case hal::CapabilityEvidenceCode::vulkan_loader_unavailable:
    return "vulkan_loader_unavailable";
  case hal::CapabilityEvidenceCode::vulkan_no_physical_device:
    return "vulkan_no_physical_device";
  case hal::CapabilityEvidenceCode::vulkan_no_compute_queue:
    return "vulkan_no_compute_queue";
  case hal::CapabilityEvidenceCode::vulkan_only_cpu_devices_available:
    return "vulkan_only_cpu_devices_available";
  case hal::CapabilityEvidenceCode::vulkan_requested_device_not_found:
    return "vulkan_requested_device_not_found";
  case hal::CapabilityEvidenceCode::vulkan_requested_device_no_compute_queue:
    return "vulkan_requested_device_no_compute_queue";
  case hal::CapabilityEvidenceCode::vulkan_requested_device_ambiguous:
    return "vulkan_requested_device_ambiguous";
  case hal::CapabilityEvidenceCode::vulkan_device_create_failed:
    return "vulkan_device_create_failed";
  case hal::CapabilityEvidenceCode::vulkan_context_uninitialized:
    return "vulkan_context_uninitialized";
  case hal::CapabilityEvidenceCode::vulkan_device_lost:
    return "vulkan_device_lost";
  case hal::CapabilityEvidenceCode::vulkan_production_hardware_not_ready:
    return "vulkan_production_hardware_not_ready";
  case hal::CapabilityEvidenceCode::open_vulkan_utility_kernels_unavailable:
    return "open_vulkan_utility_kernels_unavailable";
  case hal::CapabilityEvidenceCode::open_vulkan_runtime_diagnostics_unavailable:
    return "open_vulkan_runtime_diagnostics_unavailable";
  case hal::CapabilityEvidenceCode::open_vulkan_matting_unavailable:
    return "open_vulkan_matting_unavailable";
  case hal::CapabilityEvidenceCode::open_vulkan_matting_adapter_unavailable:
    return "open_vulkan_matting_adapter_unavailable";
  case hal::CapabilityEvidenceCode::open_vulkan_eye_contact_runtime_unavailable:
    return "open_vulkan_eye_contact_runtime_unavailable";
  case hal::CapabilityEvidenceCode::
      open_vulkan_video_noise_removal_runtime_unavailable:
    return "open_vulkan_video_noise_removal_runtime_unavailable";
  case hal::CapabilityEvidenceCode::vulkan_auto_frame_yunet_unavailable:
    return "vulkan_auto_frame_yunet_unavailable";
  case hal::CapabilityEvidenceCode::
      vulkan_vignette_tracked_center_not_supported:
    return "vulkan_vignette_tracked_center_not_supported";
  case hal::CapabilityEvidenceCode::vulkan_production_evidence_inconsistent:
    return "vulkan_production_evidence_inconsistent";
  case hal::CapabilityEvidenceCode::per_effect_capability_evidence_missing:
    return "per_effect_capability_evidence_missing";
  case hal::CapabilityEvidenceCode::open_vulkan_eye_contact_unavailable:
    return "open_vulkan_eye_contact_unavailable";
  case hal::CapabilityEvidenceCode::open_vulkan_video_noise_removal_unavailable:
    return "open_vulkan_video_noise_removal_unavailable";
  case hal::CapabilityEvidenceCode::vulkan_effect_cpu_tail:
    return "vulkan_effect_cpu_tail";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
LaneToken(video::ProductCapabilityLaneId value) noexcept {
  switch (value) {
  case video::ProductCapabilityLaneId::invalid:
    return {};
  case video::ProductCapabilityLaneId::maxine:
    return "maxine";
  case video::ProductCapabilityLaneId::open_cuda:
    return "open_cuda";
  case video::ProductCapabilityLaneId::open_vulkan:
    return "open_vulkan";
  case video::ProductCapabilityLaneId::cpu:
    return "cpu";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
DispositionToken(video::ProductCapabilityDisposition value) noexcept {
  switch (value) {
  case video::ProductCapabilityDisposition::invalid:
    return "invalid";
  case video::ProductCapabilityDisposition::production_usable:
    return "production_usable";
  case video::ProductCapabilityDisposition::usable_with_degraded_behavior:
    return "usable_with_degraded_behavior";
  case video::ProductCapabilityDisposition::experimental:
    return "experimental";
  case video::ProductCapabilityDisposition::diagnostics_only:
    return "diagnostics_only";
  case video::ProductCapabilityDisposition::unavailable:
    return "unavailable";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
RefreshToken(video::CapabilityRefreshStatus value) noexcept {
  switch (value) {
  case video::CapabilityRefreshStatus::never:
    return "never";
  case video::CapabilityRefreshStatus::in_progress:
    return "in_progress";
  case video::CapabilityRefreshStatus::succeeded:
    return "succeeded";
  case video::CapabilityRefreshStatus::failed:
    return "failed";
  case video::CapabilityRefreshStatus::stale:
    return "stale";
  }
  return "never";
}

[[nodiscard]] constexpr std::string_view
LifecycleToken(video::VideoRuntimeLifecycle value) noexcept {
  switch (value) {
  case video::VideoRuntimeLifecycle::disabled:
    return "disabled";
  case video::VideoRuntimeLifecycle::idle:
    return "idle";
  case video::VideoRuntimeLifecycle::starting:
    return "starting";
  case video::VideoRuntimeLifecycle::running:
    return "running";
  case video::VideoRuntimeLifecycle::stopping:
    return "stopping";
  case video::VideoRuntimeLifecycle::failed:
    return "failed";
  }
  return "disabled";
}

[[nodiscard]] constexpr std::string_view
StageExecutionToken(hal::StageExecutionStatus value) noexcept {
  switch (value) {
  case hal::StageExecutionStatus::not_run:
    return "not_run";
  case hal::StageExecutionStatus::succeeded:
    return "succeeded";
  case hal::StageExecutionStatus::optional_failed_open:
    return "optional_failed_open";
  case hal::StageExecutionStatus::required_failed:
    return "required_failed";
  case hal::StageExecutionStatus::disabled_preserved:
    return "disabled_preserved";
  }
  return "not_run";
}

[[nodiscard]] constexpr std::string_view
CpuTailToken(hal::CpuTailId value) noexcept {
  switch (value) {
  case hal::CpuTailId::invalid:
    return {};
  case hal::CpuTailId::output_format_conversion:
    return "output_format_conversion";
  case hal::CpuTailId::auto_frame_tracking:
    return "auto_frame_tracking";
  case hal::CpuTailId::provider_postprocess:
    return "provider_postprocess";
  case hal::CpuTailId::writer_copy:
    return "writer_copy";
  }
  return {};
}

void AppendCode(std::ostream &out, EffectCapabilityCode code) {
  out << "{\"id\":" << static_cast<std::uint32_t>(code) << ",\"token\":";
  AppendJsonString(out, video::effects::EffectCapabilityCodeToken(code));
  out << '}';
}

void AppendGenerations(std::ostream &out,
                       const video::ProductCapabilityGenerations &value) {
  out << "{\"capability\":" << value.capability
      << ",\"configuration\":" << value.configuration
      << ",\"provider\":" << value.provider << ",\"device\":" << value.device
      << ",\"context\":" << value.context << '}';
}

void AppendProgramGenerations(std::ostream &out,
                              const hal::ProgramGenerations &value) {
  out << "{\"configuration\":" << value.configuration
      << ",\"geometry\":" << value.geometry << ",\"model\":" << value.model
      << ",\"provider\":" << value.provider << ",\"device\":" << value.device
      << ",\"context\":" << value.context << '}';
}

void AppendInvalidationKey(std::ostream &out,
                           const hal::ProgramInvalidationKey &value) {
  out << "{\"generations\":";
  AppendProgramGenerations(out, value.generations);
  out << ",\"canonical_plan_hash\":" << value.canonical_plan_hash
      << ",\"model_artifact_hash\":" << value.model_artifact_hash
      << ",\"execution_domain_hash\":" << value.execution_domain_hash << '}';
}

void AppendExecutionDomain(std::ostream &out,
                           const hal::ExecutionDomainId &value) {
  out << "{\"backend\":";
  AppendJsonString(out, BackendToken(value.backend));
  out << ",\"backend_id\":" << static_cast<std::uint32_t>(value.backend)
      << ",\"api_family\":" << static_cast<std::uint32_t>(value.api_family)
      << ",\"physical_device\":{\"vendor_id\":"
      << value.physical_device.vendor_id
      << ",\"device_id\":" << value.physical_device.device_id
      << ",\"stable_id\":\"";
  const auto flags = out.flags();
  const auto fill = out.fill();
  for (const std::uint8_t byte : value.physical_device.stable_id) {
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned int>(byte);
  }
  out.flags(flags);
  out.fill(fill);
  out << "\"},\"logical_context_generation\":"
      << value.logical_context_generation << ",\"allocator_family\":"
      << static_cast<std::uint32_t>(value.allocator_family)
      << ",\"memory_handle_family\":"
      << static_cast<std::uint32_t>(value.memory_handle_family)
      << ",\"queue_or_stream_identity\":" << value.queue_or_stream_identity
      << ",\"ownership_model\":"
      << static_cast<std::uint32_t>(value.ownership_model)
      << ",\"synchronization_protocol\":"
      << static_cast<std::uint32_t>(value.synchronization_protocol)
      << ",\"adapter_instance_id\":" << value.adapter_instance_id << '}';
}

void AppendCapabilityEvidence(std::ostream &out,
                              const hal::CapabilityEvidenceRecord &value) {
  out << "{\"code_id\":" << static_cast<std::uint32_t>(value.code)
      << ",\"code\":";
  AppendJsonString(out, EvidenceToken(value.code));
  out << ",\"status_id\":" << static_cast<std::uint32_t>(value.status)
      << ",\"status\":";
  AppendJsonString(out, StatusToken(value.status));
  out << ",\"reason_id\":" << static_cast<std::uint32_t>(value.reason)
      << ",\"reason\":";
  AppendJsonString(out, ReasonToken(value.reason));
  out << ",\"effect\":";
  AppendJsonString(out, hal::StableEffectId(value.semantic));
  out << ",\"backend\":";
  AppendJsonString(out, BackendToken(value.backend));
  out << ",\"satisfied\":" << (value.satisfied ? "true" : "false") << '}';
}

void AppendEvidenceArray(
    std::ostream &out,
    const std::array<hal::CapabilityEvidenceRecord,
                     hal::kMaxCapabilityEvidenceRecords> &evidence,
    std::size_t count) {
  out << '[';
  for (std::size_t i = 0; i < count; ++i) {
    if (i != 0)
      out << ',';
    AppendCapabilityEvidence(out, evidence[i]);
  }
  out << ']';
}

void AppendCapabilityLane(std::ostream &out,
                          const video::ProductCapabilitySnapshot &snapshot,
                          const video::ProductCapabilityLane &lane) {
  out << "{\"effect\":";
  AppendJsonString(out, hal::StableEffectId(lane.effect));
  out << ",\"effect_id\":" << static_cast<std::uint32_t>(lane.effect)
      << ",\"lane\":";
  AppendJsonString(out, LaneToken(lane.lane));
  out << ",\"lane_id\":" << static_cast<unsigned int>(lane.lane)
      << ",\"disposition\":";
  AppendJsonString(out, DispositionToken(lane.disposition));
  out << ",\"code\":";
  AppendCode(out, lane.code);
  out << ",\"nested_code\":";
  AppendCode(out, lane.nested_code);
  out << ",\"selectable\":"
      << (video::IsProductCapabilityLaneEligible(snapshot, lane) ? "true"
                                                                 : "false")
      << ",\"semantic\":{\"provider\":";
  AppendJsonString(out, ProviderToken(lane.semantic.provider));
  out << ",\"provider_id\":"
      << static_cast<std::uint32_t>(lane.semantic.provider) << ",\"status\":";
  AppendJsonString(out, StatusToken(lane.semantic.status));
  out << ",\"status_id\":" << static_cast<std::uint32_t>(lane.semantic.status)
      << ",\"reason\":";
  AppendJsonString(out, ReasonToken(lane.semantic.reason));
  out << ",\"reason_id\":" << static_cast<std::uint32_t>(lane.semantic.reason)
      << ",\"live_stage\":" << (lane.semantic.live_stage ? "true" : "false")
      << ",\"production_eligible\":"
      << (lane.semantic.production_eligible ? "true" : "false")
      << ",\"exact_evidence\":"
      << (lane.semantic.exact_evidence ? "true" : "false") << ",\"cpu_tail\":";
  AppendJsonString(out, CpuTailToken(lane.semantic.cpu_tail));
  out << ",\"cpu_tail_scratch_bytes\":" << lane.semantic.cpu_tail_scratch_bytes
      << ",\"supports_tracked_vignette_center\":"
      << (lane.semantic.supports_tracked_vignette_center ? "true" : "false")
      << ",\"evidence\":";
  AppendEvidenceArray(out, lane.semantic.evidence,
                      lane.semantic.evidence_count);
  out << "}}";
}

void AppendCapabilitySnapshot(
    std::ostream &out, const video::ProductCapabilitySnapshot &snapshot) {
  out << "{\"valid\":true,\"schema_version\":" << snapshot.schema_version
      << ",\"hal_contract_version\":" << snapshot.hal_contract_version
      << ",\"reason_code_version\":" << snapshot.reason_code_version
      << ",\"generations\":";
  AppendGenerations(out, snapshot.generations);
  out << ",\"refresh\":{\"status\":";
  AppendJsonString(out, RefreshToken(snapshot.refresh.status));
  out << ",\"status_id\":" << static_cast<unsigned int>(snapshot.refresh.status)
      << ",\"code\":";
  AppendCode(out, snapshot.refresh.code);
  out << ",\"requested_generation\":" << snapshot.refresh.requested_generation
      << ",\"completed_generation\":" << snapshot.refresh.completed_generation
      << ",\"current\":"
      << (video::IsCapabilitySnapshotCurrent(snapshot) ? "true" : "false")
      << "},\"lanes\":[";
  for (std::size_t i = 0; i < snapshot.lane_count; ++i) {
    if (i != 0)
      out << ',';
    AppendCapabilityLane(out, snapshot, snapshot.lanes[i]);
  }
  out << "]}";
}

[[nodiscard]] const video::ProductCapabilityLane *
FindLane(const video::ProductCapabilitySnapshot &snapshot,
         hal::SemanticEffectId effect,
         video::ProductCapabilityLaneId lane) noexcept {
  return video::FindProductCapabilityLane(snapshot, effect, lane);
}

[[nodiscard]] bool
HasSatisfiedEvidence(const video::ProductCapabilityLane *lane,
                     hal::CapabilityEvidenceCode code) noexcept {
  if (!lane || lane->semantic.evidence_count > lane->semantic.evidence.size())
    return false;
  for (std::size_t i = 0; i < lane->semantic.evidence_count; ++i) {
    const auto &evidence = lane->semantic.evidence[i];
    if (evidence.code == code && evidence.satisfied &&
        evidence.status == hal::StatusId::ready &&
        evidence.reason == hal::ReasonId::none) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::string_view
RefreshBlocker(const video::ProductCapabilitySnapshot &snapshot) noexcept {
  if (video::IsCapabilitySnapshotCurrent(snapshot))
    return {};
  return video::effects::EffectCapabilityCodeToken(snapshot.refresh.code);
}

[[nodiscard]] std::string_view
FirstEngineBlocker(const video::ProductCapabilitySnapshot &snapshot,
                   video::ProductCapabilityLaneId lane_id) noexcept {
  if (const auto refresh = RefreshBlocker(snapshot); !refresh.empty())
    return refresh;
  for (const auto effect : video::kCanonicalVideoEffects) {
    const auto *lane = FindLane(snapshot, effect, lane_id);
    if (lane && !video::IsProductCapabilityLaneEligible(snapshot, *lane))
      return video::effects::EffectCapabilityCodeToken(lane->code);
  }
  return {};
}

void AppendAvailableEffects(std::ostream &out,
                            const video::ProductCapabilitySnapshot &snapshot,
                            video::ProductCapabilityLaneId lane_id) {
  out << '[';
  bool first = true;
  for (const auto effect : video::kCanonicalVideoEffects) {
    const auto *lane = FindLane(snapshot, effect, lane_id);
    if (!lane || !video::IsProductCapabilityLaneEligible(snapshot, *lane))
      continue;
    if (!first)
      out << ',';
    first = false;
    AppendJsonString(out, hal::StableEffectId(effect));
  }
  out << ']';
}

void AppendBlockedEffects(std::ostream &out,
                          const video::ProductCapabilitySnapshot &snapshot,
                          video::ProductCapabilityLaneId lane_id) {
  const std::string_view refresh = RefreshBlocker(snapshot);
  out << '{';
  bool first = true;
  for (const auto effect : video::kCanonicalVideoEffects) {
    const auto *lane = FindLane(snapshot, effect, lane_id);
    if (!lane || video::IsProductCapabilityLaneEligible(snapshot, *lane))
      continue;
    if (!first)
      out << ',';
    first = false;
    AppendJsonString(out, hal::StableEffectId(effect));
    out << ':';
    AppendJsonString(out,
                     refresh.empty()
                         ? video::effects::EffectCapabilityCodeToken(lane->code)
                         : refresh);
  }
  out << '}';
}

void AppendGenericEngine(std::ostream &out,
                         const video::ProductCapabilitySnapshot &snapshot,
                         video::ProductCapabilityLaneId lane_id,
                         std::uint64_t status_generation) {
  bool any_provider = false;
  bool any_eligible = false;
  for (const auto effect : video::kCanonicalVideoEffects) {
    const auto *lane = FindLane(snapshot, effect, lane_id);
    if (!lane)
      continue;
    any_provider |=
        lane->semantic.provider != hal::SemanticEffectProviderId::invalid;
    any_eligible |= video::IsProductCapabilityLaneEligible(snapshot, *lane);
  }
  out << "{\"status_generation\":" << status_generation
      << ",\"compiled_enabled\":" << (any_provider ? "true" : "false")
      << ",\"ok\":" << (any_eligible ? "true" : "false")
      << ",\"supported\":" << (any_eligible ? "true" : "false")
      << ",\"refresh_status\":";
  AppendJsonString(out, RefreshToken(snapshot.refresh.status));
  out << ",\"refresh_code\":";
  AppendJsonString(
      out, video::effects::EffectCapabilityCodeToken(snapshot.refresh.code));
  out << ",\"capability_generation\":" << snapshot.generations.capability
      << ",\"available_effects\":";
  AppendAvailableEffects(out, snapshot, lane_id);
  out << ",\"blocked_effects\":";
  AppendBlockedEffects(out, snapshot, lane_id);
  out << ",\"blocked_reason\":";
  AppendJsonString(out, FirstEngineBlocker(snapshot, lane_id));
  out << ",\"installed_models\":[],\"default_model_id\":\"\""
      << ",\"models\":[],\"missing_models\":{}";
  if (lane_id == video::ProductCapabilityLaneId::open_cuda) {
    out << ",\"onnxruntime_version\":\"\""
        << ",\"onnxruntime_providers\":[]"
        << ",\"onnxruntime_cuda_provider_present\":"
        << (any_eligible ? "true" : "false")
        << ",\"onnxruntime_tensorrt_provider_present\":false"
        << ",\"onnxruntime_cpu_provider_present\":false"
        << ",\"cuda_driver_api_available\":"
        << (any_eligible ? "true" : "false")
        << ",\"cuda_context_available\":" << (any_eligible ? "true" : "false");
  }
  out << '}';
}

void AppendOpenVulkanEngine(std::ostream &out,
                            const video::ProductCapabilitySnapshot &snapshot,
                            std::uint64_t status_generation) {
  using Evidence = hal::CapabilityEvidenceCode;
  const auto *mirror = FindLane(snapshot, hal::SemanticEffectId::mirror,
                                video::ProductCapabilityLaneId::open_vulkan);
  const auto *vignette = FindLane(snapshot, hal::SemanticEffectId::vignette,
                                  video::ProductCapabilityLaneId::open_vulkan);
  const auto *matting =
      FindLane(snapshot, hal::SemanticEffectId::virtual_background_blur,
               video::ProductCapabilityLaneId::open_vulkan);
  const auto *auto_frame =
      FindLane(snapshot, hal::SemanticEffectId::auto_frame,
               video::ProductCapabilityLaneId::open_vulkan);
  const auto *eye = FindLane(snapshot, hal::SemanticEffectId::eye_contact,
                             video::ProductCapabilityLaneId::open_vulkan);
  const auto *denoise =
      FindLane(snapshot, hal::SemanticEffectId::video_noise_removal,
               video::ProductCapabilityLaneId::open_vulkan);
  const bool current = video::IsCapabilitySnapshotCurrent(snapshot);
  const bool mirror_ready =
      mirror && video::IsProductCapabilityLaneEligible(snapshot, *mirror);
  const bool vignette_ready =
      vignette && video::IsProductCapabilityLaneEligible(snapshot, *vignette);
  const auto common = [&](Evidence code) {
    return HasSatisfiedEvidence(mirror, code) ||
           HasSatisfiedEvidence(vignette, code) ||
           HasSatisfiedEvidence(matting, code);
  };
  const bool build = common(Evidence::vulkan_build_enabled);
  const bool loader = common(Evidence::vulkan_loader_available);
  const bool physical = common(Evidence::vulkan_physical_device_available);
  const bool non_cpu = common(Evidence::vulkan_non_cpu_device);
  const bool queue = common(Evidence::vulkan_compute_queue_available);
  const bool context = common(Evidence::vulkan_context_healthy);
  const bool kernels = common(Evidence::vulkan_utility_kernels_ready);
  const bool hardware_ready =
      build && loader && physical && non_cpu && queue && context;

  out << "{\"status_generation\":" << status_generation
      << ",\"compiled_enabled\":" << (build ? "true" : "false")
      << ",\"ok\":" << ((mirror_ready || vignette_ready) ? "true" : "false")
      << ",\"supported\":"
      << ((mirror_ready || vignette_ready) ? "true" : "false")
      << ",\"runtime_library_found\":" << (loader ? "true" : "false")
      << ",\"instance_created\":" << (context ? "true" : "false")
      << ",\"physical_device_found\":" << (physical ? "true" : "false")
      << ",\"non_cpu_device_selected\":" << (non_cpu ? "true" : "false")
      << ",\"cpu_device_selected\":false"
      << ",\"compute_queue_available\":" << (queue ? "true" : "false")
      << ",\"logical_device_created\":" << (context ? "true" : "false")
      << ",\"context_created\":" << (context ? "true" : "false")
      << ",\"context_healthy\":" << (context ? "true" : "false")
      << ",\"production_hardware_ready\":"
      << (hardware_ready ? "true" : "false")
      << ",\"shader_pipeline_created\":" << (kernels ? "true" : "false")
      << ",\"refresh_status\":";
  AppendJsonString(out, RefreshToken(snapshot.refresh.status));
  out << ",\"refresh_code\":";
  AppendJsonString(
      out, video::effects::EffectCapabilityCodeToken(snapshot.refresh.code));
  out << ",\"capability_generation\":" << snapshot.generations.capability
      << ",\"refresh_requested_generation\":"
      << snapshot.refresh.requested_generation
      << ",\"refresh_completed_generation\":"
      << snapshot.refresh.completed_generation
      << ",\"refresh_current\":" << (current ? "true" : "false")
      << ",\"installed_models\":[],\"default_model_id\":\"\""
      << ",\"models\":[],\"missing_models\":{}"
      << ",\"available_effects\":";
  AppendAvailableEffects(out, snapshot,
                         video::ProductCapabilityLaneId::open_vulkan);
  out << ",\"blocked_effects\":";
  AppendBlockedEffects(out, snapshot,
                       video::ProductCapabilityLaneId::open_vulkan);
  out << ",\"blocked_reason\":";
  AppendJsonString(
      out, FirstEngineBlocker(snapshot,
                              video::ProductCapabilityLaneId::open_vulkan));
  out << ",\"mirror_production_ready\":" << (mirror_ready ? "true" : "false")
      << ",\"mirror_readiness_code\":";
  AppendJsonString(out,
                   mirror_ready
                       ? video::effects::EffectCapabilityCodeToken(mirror->code)
                       : std::string_view{});
  out << ",\"mirror_blocker_code\":";
  AppendJsonString(out, mirror_ready
                            ? std::string_view{}
                            : (!RefreshBlocker(snapshot).empty()
                                   ? RefreshBlocker(snapshot)
                                   : video::effects::EffectCapabilityCodeToken(
                                         mirror->code)));
  out << ",\"vignette_fixed_center_production_ready\":"
      << (vignette_ready ? "true" : "false") << ",\"vignette_readiness_code\":";
  AppendJsonString(
      out, vignette_ready
               ? video::effects::EffectCapabilityCodeToken(vignette->code)
               : std::string_view{});
  out << ",\"vignette_blocker_code\":";
  AppendJsonString(out, vignette_ready
                            ? std::string_view{}
                            : (!RefreshBlocker(snapshot).empty()
                                   ? RefreshBlocker(snapshot)
                                   : video::effects::EffectCapabilityCodeToken(
                                         vignette->code)));
  out << ",\"vignette_parameter_contract\":\"fixed_center\""
      << ",\"auto_frame_production_ready\":false"
      << ",\"auto_frame_crop_stage_implemented\":"
      << ((auto_frame && auto_frame->semantic.live_stage) ? "true" : "false")
      << ",\"auto_frame_cpu_tail\":"
      << ((auto_frame &&
           (auto_frame->nested_code ==
                EffectCapabilityCode::vulkan_effect_cpu_tail ||
            auto_frame->semantic.cpu_tail != hal::CpuTailId::invalid))
              ? "true"
              : "false")
      << ",\"auto_frame_selectable_cpu_fallback\":false"
      << ",\"auto_frame_readiness_code\":\"\""
      << ",\"auto_frame_degraded_reason_code\":\"vulkan_effect_cpu_tail\""
      << ",\"auto_frame_blocker_code\":";
  AppendJsonString(out, auto_frame ? video::effects::EffectCapabilityCodeToken(
                                         auto_frame->code)
                                   : std::string_view{});
  out << ",\"matting_runtime\":\"\""
      << ",\"matting_production_ready\":false"
      << ",\"matting_build_enabled\":" << (build ? "true" : "false")
      << ",\"matting_adapter_available\":false"
      << ",\"matting_runtime_created\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_matting_runtime_ready)
              ? "true"
              : "false")
      << ",\"matting_graph_loaded\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_matting_runtime_ready)
              ? "true"
              : "false")
      << ",\"matting_model_pack_selected\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_model_artifacts_valid)
              ? "true"
              : "false")
      << ",\"matting_model_contract_validated\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_model_artifacts_valid)
              ? "true"
              : "false")
      << ",\"matting_warmup_complete\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_warmup_complete)
              ? "true"
              : "false")
      << ",\"input_device_resident\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_device_local_residency)
              ? "true"
              : "false")
      << ",\"alpha_device_resident\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_device_local_residency)
              ? "true"
              : "false")
      << ",\"output_device_resident\":"
      << (HasSatisfiedEvidence(matting, Evidence::vulkan_device_local_residency)
              ? "true"
              : "false")
      << ",\"device_residency_mode\":";
  AppendJsonString(out, HasSatisfiedEvidence(
                            matting, Evidence::vulkan_device_local_residency)
                            ? std::string_view{"device_resident"}
                            : std::string_view{});
  out << ",\"matting_reason_code\":";
  AppendJsonString(
      out, matting ? video::effects::EffectCapabilityCodeToken(matting->code)
                   : std::string_view{});
  out << ",\"matting_blocker_code\":";
  AppendJsonString(out, matting ? video::effects::EffectCapabilityCodeToken(
                                      matting->nested_code)
                                : std::string_view{});
  out << ",\"matting_cpu_layers_used\":false"
      << ",\"matting_shared_device_imported\":false"
      << ",\"matting_queue_ownership_explicit\":false"
      << ",\"matting_synchronous_completion\":false"
      << ",\"matting_bounded_reusable_allocations\":false"
      << ",\"matting_persistent_allocation_count\":0"
      << ",\"matting_dynamic_allocation_count\":0"
      << ",\"matting_cpu_readback_count\":0"
      << ",\"matting_cpu_fallback_count\":0"
      << ",\"matting_parity_validated\":false"
      << ",\"eye_contact_production_ready\":false"
      << ",\"eye_contact_reason_code\":";
  AppendJsonString(out,
                   eye ? video::effects::EffectCapabilityCodeToken(eye->code)
                       : std::string_view{});
  out << ",\"eye_contact_live_stage_implemented\":"
      << ((eye && eye->semantic.live_stage) ? "true" : "false")
      << ",\"eye_contact_blocker_code\":";
  AppendJsonString(
      out, eye ? video::effects::EffectCapabilityCodeToken(eye->nested_code)
               : std::string_view{});
  out << ",\"eye_contact_production_adapter_available\":false"
      << ",\"eye_contact_vulkan_inference_provider_available\":false"
      << ",\"eye_contact_shared_device_imported\":false"
      << ",\"eye_contact_queue_ownership_explicit\":false"
      << ",\"eye_contact_model_pack_selected\":false"
      << ",\"eye_contact_artifact_contract_validated\":false"
      << ",\"eye_contact_device_resident_analysis\":false"
      << ",\"eye_contact_device_resident_tensor_io\":false"
      << ",\"eye_contact_warmup_complete\":false"
      << ",\"eye_contact_bounded_reusable_allocations\":false"
      << ",\"eye_contact_synchronization_contract_validated\":false"
      << ",\"eye_contact_parity_validated\":false"
      << ",\"eye_contact_selectable_cpu_fallback\":false"
      << ",\"eye_contact_cpu_readback_count\":0"
      << ",\"eye_contact_cpu_fallback_count\":0"
      << ",\"video_noise_removal_production_ready\":false"
      << ",\"video_noise_removal_reason_code\":";
  AppendJsonString(
      out, denoise ? video::effects::EffectCapabilityCodeToken(denoise->code)
                   : std::string_view{});
  out << ",\"video_noise_removal_live_stage_implemented\":"
      << ((denoise && denoise->semantic.live_stage) ? "true" : "false")
      << ",\"video_noise_removal_blocker_code\":";
  AppendJsonString(out, denoise ? video::effects::EffectCapabilityCodeToken(
                                      denoise->nested_code)
                                : std::string_view{});
  out << ",\"video_noise_removal_production_adapter_available\":false"
      << ",\"video_noise_removal_vulkan_inference_provider_available\":false"
      << ",\"video_noise_removal_shared_device_imported\":false"
      << ",\"video_noise_removal_queue_ownership_explicit\":false"
      << ",\"video_noise_removal_model_pack_selected\":false"
      << ",\"video_noise_removal_artifact_contract_validated\":false"
      << ",\"video_noise_removal_fully_device_resident_tensor_io\":false"
      << ",\"video_noise_removal_device_resident_preprocess\":false"
      << ",\"video_noise_removal_device_resident_postprocess\":false"
      << ",\"video_noise_removal_warmup_complete\":false"
      << ",\"video_noise_removal_synchronization_contract_validated\":false"
      << ",\"video_noise_removal_bounded_reusable_allocations\":false"
      << ",\"video_noise_removal_temporal_history_device_resident\":false"
      << ",\"video_noise_removal_temporal_history_bounded\":false"
      << ",\"video_noise_removal_history_reset_on_disable\":false"
      << ",\"video_noise_removal_history_reset_on_reconfigure\":false"
      << ",\"video_noise_removal_capture_sequence_discontinuity_reset\":false"
      << ",\"video_noise_removal_parity_validated\":false"
      << ",\"video_noise_removal_selectable_cpu_fallback\":false"
      << ",\"video_noise_removal_cpu_readback_count\":0"
      << ",\"video_noise_removal_cpu_fallback_count\":0" << '}';
}

[[nodiscard]] constexpr video::ProductCapabilityLaneId
LaneForBackend(hal::BackendId backend) noexcept {
  switch (backend) {
  case hal::BackendId::maxine:
    return video::ProductCapabilityLaneId::maxine;
  case hal::BackendId::cuda:
  case hal::BackendId::open_video_onnxruntime:
    return video::ProductCapabilityLaneId::open_cuda;
  case hal::BackendId::vulkan:
    return video::ProductCapabilityLaneId::open_vulkan;
  case hal::BackendId::cpu:
    return video::ProductCapabilityLaneId::cpu;
  case hal::BackendId::invalid:
  case hal::BackendId::test_fake:
    break;
  }
  return video::ProductCapabilityLaneId::invalid;
}

[[nodiscard]] constexpr std::string_view
LegacyEffectBackendToken(video::ProductCapabilityLaneId lane) noexcept {
  switch (lane) {
  case video::ProductCapabilityLaneId::maxine:
    return "maxine";
  case video::ProductCapabilityLaneId::open_cuda:
    return "open_cuda";
  case video::ProductCapabilityLaneId::open_vulkan:
    return "open_vulkan";
  case video::ProductCapabilityLaneId::cpu:
    return "cpu";
  case video::ProductCapabilityLaneId::invalid:
    break;
  }
  return {};
}

[[nodiscard]] const hal::ResolvedStage *
FindResolvedStage(const hal::ResolutionResult &resolution,
                  hal::SemanticEffectId effect) noexcept {
  if (resolution.stage_count > resolution.stages.size())
    return nullptr;
  for (std::size_t i = 0; i < resolution.stage_count; ++i) {
    if (resolution.stages[i].intent.semantic == effect)
      return &resolution.stages[i];
  }
  return nullptr;
}

[[nodiscard]] const video::ProductCapabilityLane *
ChooseEffectLane(const video::VideoRuntimeStatusSnapshot &snapshot,
                 hal::SemanticEffectId effect) noexcept {
  const auto *resolved = FindResolvedStage(snapshot.resolution, effect);
  if (resolved && resolved->selected_backend != hal::BackendId::invalid) {
    return FindLane(snapshot.capabilities, effect,
                    LaneForBackend(resolved->selected_backend));
  }
  if (snapshot.resolution.request_mode ==
          hal::BackendRequestMode::explicit_backend &&
      snapshot.resolution.requested_backend != hal::BackendId::invalid) {
    return FindLane(snapshot.capabilities, effect,
                    LaneForBackend(snapshot.resolution.requested_backend));
  }
  constexpr std::array precedence{
      video::ProductCapabilityLaneId::maxine,
      video::ProductCapabilityLaneId::open_cuda,
      video::ProductCapabilityLaneId::open_vulkan,
      video::ProductCapabilityLaneId::cpu,
  };
  for (const auto lane_id : precedence) {
    const auto *lane = FindLane(snapshot.capabilities, effect, lane_id);
    if (lane &&
        video::IsProductCapabilityLaneEligible(snapshot.capabilities, *lane))
      return lane;
  }
  return FindLane(snapshot.capabilities, effect,
                  video::ProductCapabilityLaneId::maxine);
}

void AppendEffectReadiness(std::ostream &out,
                           const video::VideoRuntimeStatusSnapshot &snapshot) {
  const std::string_view refresh = RefreshBlocker(snapshot.capabilities);
  out << '{';
  for (std::size_t i = 0; i < video::kCanonicalVideoEffects.size(); ++i) {
    if (i != 0)
      out << ',';
    const auto effect = video::kCanonicalVideoEffects[i];
    const auto *resolved = FindResolvedStage(snapshot.resolution, effect);
    const auto *lane = ChooseEffectLane(snapshot, effect);
    const bool eligible = lane && video::IsProductCapabilityLaneEligible(
                                      snapshot.capabilities, *lane);
    std::string_view state = eligible ? "ready" : "backend_unavailable";
    if (eligible && resolved && resolved->status == hal::StatusId::degraded)
      state = "recoverable_error";
    std::string_view reason;
    if (!refresh.empty())
      reason = refresh;
    else if (!eligible && lane)
      reason = video::effects::EffectCapabilityCodeToken(lane->code);
    else if (resolved)
      reason = ReasonToken(resolved->reason);
    const auto lane_id =
        lane ? lane->lane : video::ProductCapabilityLaneId::invalid;

    AppendJsonString(out, hal::StableEffectId(effect));
    out << ":{\"state\":";
    AppendJsonString(out, state);
    out << ",\"summary\":";
    AppendJsonString(out,
                     eligible ? "Effect is ready." : "Effect is unavailable.");
    out << ",\"detail\":";
    AppendJsonString(out, reason);
    out << ",\"backend\":";
    AppendJsonString(out, LegacyEffectBackendToken(lane_id));
    out << ",\"requested_model_id\":\"\",\"resolved_model_id\":\"\""
        << ",\"reason\":";
    AppendJsonString(out, reason);
    out << '}';
  }
  out << '}';
}

void AppendLegacyEffectsBackends(
    std::ostream &out,
    const video::LegacyVideoRuntimeCompatibility &compatibility) {
  bool first = true;
  for (const auto effect : video::kCanonicalVideoEffects) {
    for (std::size_t i = 0; i < compatibility.active_effect_count; ++i) {
      const auto &active = compatibility.active_effects[i];
      if (active.effect != effect)
        continue;
      if (!first)
        out << ',';
      first = false;
      out << active.effect_id << ':' << active.backend;
      break;
    }
  }
}

[[nodiscard]] constexpr std::string_view
RequestedBackendToken(const hal::ResolutionResult &resolution) noexcept {
  if (resolution.request_mode == hal::BackendRequestMode::automatic)
    return "auto";
  return BackendToken(resolution.requested_backend);
}

[[nodiscard]] std::string_view
FirstActiveProvider(const video::ActiveRuntimeAttribution &active) noexcept {
  if (active.effect_count == 0)
    return {};
  if (active.mixed_backends)
    return "mixed";
  return ProviderToken(active.effects[0].provider);
}

void AppendTransferBucket(std::ostream &out,
                          const hal::ExecutionTelemetry &telemetry,
                          video::ProductCapabilityLaneId bucket) {
  const bool owns = LaneForBackend(telemetry.selected_backend) == bucket;
  const auto &transfers = telemetry.transfers;
  out << "{\"active_frames\":" << (owns ? telemetry.active_backend_frames : 0)
      << ",\"uploads\":" << (owns ? transfers.uploads : 0) << ",\"downloads\":"
      << (owns ? transfers.final_readbacks + transfers.intermediate_readbacks
               : 0)
      << ",\"final_downloads\":" << (owns ? transfers.final_readbacks : 0)
      << ",\"cpu_continuation_downloads\":"
      << (owns ? transfers.intermediate_readbacks : 0)
      << ",\"alpha_downloads\":0,\"matte_frame_uploads\":0"
      << ",\"standalone_scaler_uploads\":0"
      << ",\"standalone_scaler_downloads\":0"
      << ",\"forced_syncs\":" << (owns ? transfers.synchronizations : 0) << '}';
}

void AppendVideoCompute(
    std::ostream &out, const video::VideoRuntimeStatusSnapshot &snapshot,
    const video::LegacyVideoRuntimeCompatibility &compatibility,
    const video::ActiveRuntimeAttribution &active) {
  const std::string_view requested = RequestedBackendToken(snapshot.resolution);
  const std::string_view selected = compatibility.selected_backend_token;
  const std::string_view active_backend = compatibility.active_backend_token;
  const std::string_view resolution_reason =
      ReasonToken(snapshot.resolution.reason);
  const bool fallback_active =
      snapshot.resolution.fallback && !active_backend.empty();
  out << "{\"status_generation\":" << snapshot.status_generation
      << ",\"capability_generation\":"
      << snapshot.capabilities.generations.capability
      << ",\"refresh_generation\":"
      << snapshot.capabilities.refresh.completed_generation
      << ",\"refresh_current\":"
      << (video::IsCapabilitySnapshotCurrent(snapshot.capabilities) ? "true"
                                                                    : "false")
      << ",\"preference\":";
  AppendJsonString(out, requested);
  out << ",\"resolved_backend\":";
  AppendJsonString(out, selected);
  out << ",\"active_backend\":";
  AppendJsonString(out, active_backend);
  out << ",\"fallback_reason\":";
  AppendJsonString(out, snapshot.resolution.fallback ? resolution_reason
                                                     : std::string_view{});
  out << ",\"degraded_reason\":";
  AppendJsonString(out, snapshot.resolution.degraded ? resolution_reason
                                                     : std::string_view{});
  out << ",\"active_engines\":[";
  std::array<video::ProductCapabilityLaneId, video::kProductCapabilityLaneCount>
      emitted{};
  std::size_t emitted_count = 0;
  for (std::size_t i = 0; i < active.effect_count; ++i) {
    const auto lane = LaneForBackend(active.effects[i].backend);
    bool duplicate = false;
    for (std::size_t j = 0; j < emitted_count; ++j)
      duplicate |= emitted[j] == lane;
    if (duplicate)
      continue;
    if (emitted_count != 0)
      out << ',';
    emitted[emitted_count++] = lane;
    AppendJsonString(out, LegacyEffectBackendToken(lane));
  }
  out << "],\"unavailable_reasons\":{\"cuda\":";
  AppendJsonString(
      out, FirstEngineBlocker(snapshot.capabilities,
                              video::ProductCapabilityLaneId::open_cuda));
  out << ",\"vulkan\":";
  AppendJsonString(
      out, FirstEngineBlocker(snapshot.capabilities,
                              video::ProductCapabilityLaneId::open_vulkan));
  out << ",\"maxine\":";
  AppendJsonString(out,
                   FirstEngineBlocker(snapshot.capabilities,
                                      video::ProductCapabilityLaneId::maxine));
  out << "},\"fallback\":{\"active\":" << (fallback_active ? "true" : "false")
      << ",\"from\":";
  AppendJsonString(out, requested);
  out << ",\"to\":";
  AppendJsonString(out, active_backend.empty() ? selected : active_backend);
  out << ",\"code\":";
  AppendJsonString(out,
                   fallback_active ? "backend_changed" : std::string_view{});
  out << ",\"detail\":";
  AppendJsonString(out,
                   fallback_active ? resolution_reason : std::string_view{});
  out << "},\"provider\":{\"mode\":";
  AppendJsonString(out, active_backend);
  out << ",\"active_provider\":";
  AppendJsonString(out, FirstActiveProvider(active));
  out << ",\"device\":\"\",\"tensor_io_mode\":\"\""
      << "},\"cpu_tails\":{\"active\":"
      << (snapshot.telemetry.transfers.cpu_tail_boundaries != 0 ? "true"
                                                                : "false")
      << ",\"stage_calls\":" << snapshot.telemetry.transfers.cpu_tail_boundaries
      << ",\"stages\":[],\"cuda_tensor_tail\":false"
      << ",\"counts\":{\"open_cuda\":"
      << (LaneForBackend(snapshot.telemetry.selected_backend) ==
                  video::ProductCapabilityLaneId::open_cuda
              ? snapshot.telemetry.transfers.cpu_tail_boundaries
              : 0)
      << ",\"open_vulkan\":"
      << (LaneForBackend(snapshot.telemetry.selected_backend) ==
                  video::ProductCapabilityLaneId::open_vulkan
              ? snapshot.telemetry.transfers.cpu_tail_boundaries
              : 0)
      << ",\"maxine\":"
      << (LaneForBackend(snapshot.telemetry.selected_backend) ==
                  video::ProductCapabilityLaneId::maxine
              ? snapshot.telemetry.transfers.cpu_tail_boundaries
              : 0)
      << ",\"denoise\":0}},\"transfers\":{\"open_cuda\":";
  AppendTransferBucket(out, snapshot.telemetry,
                       video::ProductCapabilityLaneId::open_cuda);
  out << ",\"open_vulkan\":";
  AppendTransferBucket(out, snapshot.telemetry,
                       video::ProductCapabilityLaneId::open_vulkan);
  out << ",\"maxine\":";
  AppendTransferBucket(out, snapshot.telemetry,
                       video::ProductCapabilityLaneId::maxine);
  out << "}}";
}

void AppendLatencyHistogram(std::ostream &out,
                            const hal::LatencyHistogram &value) {
  out << "{\"samples\":" << value.samples << ",\"total_ns\":" << value.total_ns
      << ",\"maximum_ns\":" << value.maximum_ns
      << ",\"p95_ns\":" << value.p95_ns << ",\"p99_ns\":" << value.p99_ns
      << ",\"buckets\":[";
  for (std::size_t i = 0; i < value.bucket_counts.size(); ++i) {
    if (i != 0)
      out << ',';
    out << "{\"upper_ns\":" << value.bucket_upper_ns[i]
        << ",\"count\":" << value.bucket_counts[i] << '}';
  }
  out << "]}";
}

void AppendTransferCounters(std::ostream &out,
                            const hal::TransferCounters &value) {
  out << "{\"uploads\":" << value.uploads
      << ",\"final_readbacks\":" << value.final_readbacks
      << ",\"intermediate_readbacks\":" << value.intermediate_readbacks
      << ",\"intermediate_uploads\":" << value.intermediate_uploads
      << ",\"synchronizations\":" << value.synchronizations
      << ",\"cpu_tail_boundaries\":" << value.cpu_tail_boundaries << '}';
}

void AppendTelemetry(std::ostream &out,
                     const video::VideoRuntimeStatusSnapshot &snapshot) {
  const auto &value = snapshot.telemetry;
  out << "{\"requested_backend\":";
  AppendJsonString(out, BackendToken(value.requested_backend));
  out << ",\"selected_backend\":";
  AppendJsonString(out, BackendToken(value.selected_backend));
  out << ",\"active_backend\":";
  AppendJsonString(out, BackendToken(value.active_backend));
  out << ",\"active_provider\":";
  AppendJsonString(out, ProviderToken(value.active_provider));
  out << ",\"execution_attempts\":" << value.execution_attempts
      << ",\"execution_successes\":" << value.execution_successes
      << ",\"committed_output_frames\":" << value.committed_output_frames
      << ",\"active_backend_frames\":" << value.active_backend_frames
      << ",\"fallback_frames\":" << value.fallback_frames
      << ",\"optional_fail_open_stages\":" << value.optional_fail_open_stages
      << ",\"required_stage_failures\":" << value.required_stage_failures
      << ",\"temporal_resets\":" << value.temporal_resets
      << ",\"device_losses\":" << value.device_losses
      << ",\"setup_attempts\":" << value.setup_attempts
      << ",\"setup_successes\":" << value.setup_successes
      << ",\"setup_rollbacks\":" << value.setup_rollbacks
      << ",\"setup_retention_rejections\":" << value.setup_retention_rejections
      << ",\"circuit_breaker_opens\":" << value.circuit_breaker_opens
      << ",\"disabled_stage_frames\":" << value.disabled_stage_frames
      << ",\"rebuild_requests\":" << value.rebuild_requests
      << ",\"frame_drop_events\":" << value.frame_drop_events
      << ",\"dropped_capture_frames\":" << value.dropped_capture_frames
      << ",\"preserve_order_failures\":" << value.preserve_order_failures
      << ",\"latest_frame_wins_events\":" << value.latest_frame_wins_events
      << ",\"disabled_stage_mask\":" << value.disabled_stage_mask
      << ",\"runtime_status\":";
  AppendJsonString(out, StatusToken(value.runtime_status));
  out << ",\"runtime_reason\":";
  AppendJsonString(out, ReasonToken(value.runtime_reason));
  out << ",\"transfers\":";
  AppendTransferCounters(out, value.transfers);
  out << ",\"frame_latency\":";
  AppendLatencyHistogram(out, value.frame_latency);
  out << ",\"stage_latency\":[";
  for (std::size_t i = 0; i < snapshot.resolution.stage_count; ++i) {
    if (i != 0)
      out << ',';
    out << "{\"stage_id\":" << snapshot.resolution.stages[i].intent.stage_id
        << ",\"effect\":";
    AppendJsonString(out, hal::StableEffectId(
                              snapshot.resolution.stages[i].intent.semantic));
    out << ",\"latency\":";
    AppendLatencyHistogram(out, value.stage_latency[i]);
    out << ",\"committed_successes\":" << value.committed_stage_successes[i]
        << '}';
  }
  out << "]}";
}

void AppendResolution(std::ostream &out, const hal::ResolutionResult &value) {
  out << "{\"request_mode\":";
  AppendJsonString(out, value.request_mode == hal::BackendRequestMode::automatic
                            ? std::string_view{"automatic"}
                            : std::string_view{"explicit_backend"});
  out << ",\"strict_backend_selection\":"
      << (value.strict_backend_selection ? "true" : "false") << ",\"status\":";
  AppendJsonString(out, StatusToken(value.status));
  out << ",\"reason\":";
  AppendJsonString(out, ReasonToken(value.reason));
  out << ",\"requested_backend\":";
  AppendJsonString(out, BackendToken(value.requested_backend));
  out << ",\"selected_backend\":";
  AppendJsonString(out, BackendToken(value.selected_backend));
  out << ",\"selected_provider\":";
  AppendJsonString(out, ProviderToken(value.selected_provider));
  out << ",\"available\":" << (value.available ? "true" : "false")
      << ",\"degraded\":" << (value.degraded ? "true" : "false")
      << ",\"fallback\":" << (value.fallback ? "true" : "false")
      << ",\"passthrough\":" << (value.passthrough ? "true" : "false")
      << ",\"stages\":[";
  for (std::size_t i = 0; i < value.stage_count; ++i) {
    if (i != 0)
      out << ',';
    const auto &stage = value.stages[i];
    out << "{\"stage_id\":" << stage.intent.stage_id << ",\"effect\":";
    AppendJsonString(out, hal::StableEffectId(stage.intent.semantic));
    out << ",\"required\":" << (stage.intent.required ? "true" : "false")
        << ",\"status\":";
    AppendJsonString(out, StatusToken(stage.status));
    out << ",\"reason\":";
    AppendJsonString(out, ReasonToken(stage.reason));
    out << ",\"selected_backend\":";
    AppendJsonString(out, BackendToken(stage.selected_backend));
    out << ",\"provider\":";
    AppendJsonString(out, ProviderToken(stage.provider));
    out << ",\"adapter_instance_id\":" << stage.adapter_instance_id
        << ",\"execution_domain\":";
    AppendExecutionDomain(out, stage.execution_domain);
    out << ",\"cpu_tail\":";
    AppendJsonString(out, CpuTailToken(stage.cpu_tail));
    out << ",\"cpu_tail_scratch_bytes\":" << stage.cpu_tail_scratch_bytes
        << ",\"skipped_optional\":"
        << (stage.skipped_optional ? "true" : "false")
        << ",\"fallback_selected\":"
        << (stage.fallback_selected ? "true" : "false") << ",\"evidence\":";
    AppendEvidenceArray(out, stage.evidence, stage.evidence_count);
    out << '}';
  }
  out << "]}";
}

void AppendReadiness(std::ostream &out,
                     const video::TypedRuntimeReadiness &readiness) {
  out << "{\"configured\":" << (readiness.configured ? "true" : "false")
      << ",\"program_id\":" << readiness.program_id << ",\"invalidation_key\":";
  AppendInvalidationKey(out, readiness.invalidation_key);
  out << ",\"status\":";
  AppendJsonString(out, StatusToken(readiness.value.status));
  out << ",\"reason\":";
  AppendJsonString(out, ReasonToken(readiness.value.reason));
  out << ",\"requested_backend\":";
  AppendJsonString(out, BackendToken(readiness.value.requested_backend));
  out << ",\"selected_backend\":";
  AppendJsonString(out, BackendToken(readiness.value.selected_backend));
  out << ",\"provider\":";
  AppendJsonString(out, ProviderToken(readiness.value.provider));
  out << ",\"execution_domain\":";
  AppendExecutionDomain(out, readiness.value.execution_domain);
  out << ",\"ready_stage_mask\":" << readiness.value.ready_stage_mask
      << ",\"blocked_stage_mask\":" << readiness.value.blocked_stage_mask
      << ",\"disabled_stage_mask\":" << readiness.value.disabled_stage_mask
      << ",\"available\":" << (readiness.value.available ? "true" : "false")
      << ",\"degraded\":" << (readiness.value.degraded ? "true" : "false")
      << ",\"rebuild_requested\":"
      << (readiness.value.rebuild_requested ? "true" : "false")
      << ",\"evidence\":";
  AppendEvidenceArray(out, readiness.value.evidence,
                      readiness.value.evidence_count);
  out << '}';
}

void AppendCommittedFrame(std::ostream &out,
                          const hal::CommittedFrameSnapshot &value) {
  out << "{\"valid\":" << (value.valid ? "true" : "false")
      << ",\"program_id\":" << value.program_id << ",\"invalidation_key\":";
  AppendInvalidationKey(out, value.invalidation_key);
  out << ",\"execution_domain\":";
  AppendExecutionDomain(out, value.execution_domain);
  out << ",\"capture_sequence\":" << value.capture_sequence
      << ",\"output_commit_sequence\":" << value.output_commit_sequence
      << ",\"stages\":[";
  for (std::size_t i = 0; i < value.stage_count; ++i) {
    if (i != 0)
      out << ',';
    const auto &stage = value.stages[i];
    out << "{\"stage_id\":" << stage.stage_id << ",\"effect\":";
    AppendJsonString(out, hal::StableEffectId(stage.semantic));
    out << ",\"primitive_id\":" << static_cast<std::uint32_t>(stage.primitive)
        << ",\"backend\":";
    AppendJsonString(out, BackendToken(stage.backend));
    out << ",\"provider\":";
    AppendJsonString(out, ProviderToken(stage.provider));
    out << ",\"status\":";
    AppendJsonString(out, StageExecutionToken(stage.status));
    out << ",\"reason\":";
    AppendJsonString(out, ReasonToken(stage.reason));
    out << '}';
  }
  out << "]}";
}

void AppendActiveAttribution(std::ostream &out,
                             const video::ActiveRuntimeAttribution &active) {
  out << "{\"active_backend\":";
  AppendJsonString(out, active.mixed_backends
                            ? std::string_view{"mixed"}
                            : BackendToken(active.active_backend));
  out << ",\"mixed_backends\":" << (active.mixed_backends ? "true" : "false")
      << ",\"effects\":[";
  for (std::size_t i = 0; i < active.effect_count; ++i) {
    if (i != 0)
      out << ',';
    const auto &effect = active.effects[i];
    out << "{\"stage_id\":" << effect.stage_id << ",\"effect\":";
    AppendJsonString(out, hal::StableEffectId(effect.effect));
    out << ",\"backend\":";
    AppendJsonString(out, BackendToken(effect.backend));
    out << ",\"provider\":";
    AppendJsonString(out, ProviderToken(effect.provider));
    out << '}';
  }
  out << "]}";
}

void AppendTypedRuntime(std::ostream &out,
                        const video::VideoRuntimeStatusSnapshot &snapshot,
                        const video::ActiveRuntimeAttribution &active) {
  out << "{\"valid\":true,\"schema_version\":" << snapshot.schema_version
      << ",\"hal_contract_version\":" << snapshot.hal_contract_version
      << ",\"status_generation\":" << snapshot.status_generation
      << ",\"generations\":";
  AppendGenerations(out, snapshot.generations);
  out << ",\"lifecycle\":";
  AppendJsonString(out, LifecycleToken(snapshot.lifecycle));
  out << ",\"capabilities\":";
  AppendCapabilitySnapshot(out, snapshot.capabilities);
  out << ",\"resolution\":";
  AppendResolution(out, snapshot.resolution);
  out << ",\"readiness\":";
  AppendReadiness(out, snapshot.readiness);
  out << ",\"latest_committed_frame\":";
  AppendCommittedFrame(out, snapshot.latest_committed_frame);
  out << ",\"active\":";
  AppendActiveAttribution(out, active);
  out << ",\"telemetry\":";
  AppendTelemetry(out, snapshot);
  out << '}';
}

} // namespace

ProductCapabilityJsonResult SerializeProductCapabilityStatusJson(
    const video::ProductCapabilitySnapshot &snapshot) {
  ProductCapabilityJsonResult result;
  result.validation = video::ValidateProductCapabilitySnapshot(snapshot);
  if (result.validation.code != video::CapabilityValidationCode::valid)
    return result;
  std::ostringstream out;
  AppendCapabilitySnapshot(out, snapshot);
  result.json = out.str();
  result.ok = true;
  return result;
}

VideoRuntimeStatusJsonResult SerializeVideoRuntimeStatusJson(
    const video::VideoRuntimeStatusSnapshot &snapshot) {
  VideoRuntimeStatusJsonResult result;
  result.validation = video::ValidateVideoRuntimeStatusSnapshot(snapshot);
  if (result.validation.code != video::RuntimeStatusValidationCode::valid)
    return result;

  const auto active = video::DeriveActiveRuntimeAttribution(snapshot);
  const auto compatibility =
      video::BuildLegacyVideoRuntimeCompatibility(snapshot);
  std::ostringstream out;
  out << "{\"typed_video_status\":";
  AppendTypedRuntime(out, snapshot, active);
  out << ",\"engines\":{\"maxine\":";
  AppendGenericEngine(out, snapshot.capabilities,
                      video::ProductCapabilityLaneId::maxine,
                      snapshot.status_generation);
  out << ",\"open_cuda\":";
  AppendGenericEngine(out, snapshot.capabilities,
                      video::ProductCapabilityLaneId::open_cuda,
                      snapshot.status_generation);
  out << ",\"open_vulkan\":";
  AppendOpenVulkanEngine(out, snapshot.capabilities,
                         snapshot.status_generation);
  out << "},\"video\":{\"compute\":";
  AppendVideoCompute(out, snapshot, compatibility, active);
  out << ",\"effect_readiness\":";
  AppendEffectReadiness(out, snapshot);
  out << ",\"pipeline\":{\"status_generation\":" << snapshot.status_generation
      << ",\"running\":"
      << (snapshot.lifecycle == video::VideoRuntimeLifecycle::running ? "true"
                                                                      : "false")
      << ",\"starting\":"
      << (snapshot.lifecycle == video::VideoRuntimeLifecycle::starting
              ? "true"
              : "false")
      << ",\"state\":";
  AppendJsonString(out, LifecycleToken(snapshot.lifecycle));
  out << ",\"effects_backends\":\"";
  AppendLegacyEffectsBackends(out, compatibility);
  out << "\"}}}";
  result.json = out.str();
  result.ok = true;
  return result;
}

} // namespace studiocast::daemon
