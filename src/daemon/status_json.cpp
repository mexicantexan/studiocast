#include "daemon/status_json.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <limits>
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

[[nodiscard]] constexpr std::string_view
PreparedRouteToken(video::PreparedHardwareRoute value) noexcept {
  switch (value) {
  case video::PreparedHardwareRoute::legacy_shadow:
    return "legacy_shadow";
  case video::PreparedHardwareRoute::hal_live:
    return "hal_live";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
PreparedEligibilityToken(video::PreparedHardwareEligibility value) noexcept {
  switch (value) {
  case video::PreparedHardwareEligibility::eligible:
    return "eligible";
  case video::PreparedHardwareEligibility::not_strict_explicit_vulkan:
    return "not_strict_explicit_vulkan";
  case video::PreparedHardwareEligibility::no_qualified_final_stage:
    return "no_qualified_final_stage";
  case video::PreparedHardwareEligibility::contains_unqualified_stage:
    return "contains_unqualified_stage";
  case video::PreparedHardwareEligibility::tracked_vignette_center:
    return "tracked_vignette_center";
  case video::PreparedHardwareEligibility::unsupported_host_layout:
    return "unsupported_host_layout";
  case video::PreparedHardwareEligibility::invalid_generation:
    return "invalid_generation";
  case video::PreparedHardwareEligibility::unavailable:
    return "unavailable";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
SetupModeToken(video::HardwareExecutionSetupMode value) noexcept {
  switch (value) {
  case video::HardwareExecutionSetupMode::hal_live:
    return "hal_live";
  case video::HardwareExecutionSetupMode::shadow_only:
    return "shadow_only";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
ApiFamilyToken(hal::ApiFamily value) noexcept {
  switch (value) {
  case hal::ApiFamily::invalid:
    return "invalid";
  case hal::ApiFamily::host:
    return "host";
  case hal::ApiFamily::cuda_driver:
    return "cuda_driver";
  case hal::ApiFamily::maxine_nvcv:
    return "maxine_nvcv";
  case hal::ApiFamily::onnxruntime:
    return "onnxruntime";
  case hal::ApiFamily::vulkan:
    return "vulkan";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
AllocatorFamilyToken(hal::AllocatorFamily value) noexcept {
  switch (value) {
  case hal::AllocatorFamily::invalid:
    return "invalid";
  case hal::AllocatorFamily::host:
    return "host";
  case hal::AllocatorFamily::cuda_driver:
    return "cuda_driver";
  case hal::AllocatorFamily::maxine_nvcv:
    return "maxine_nvcv";
  case hal::AllocatorFamily::onnxruntime:
    return "onnxruntime";
  case hal::AllocatorFamily::vulkan:
    return "vulkan";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
MemoryHandleFamilyToken(hal::MemoryHandleFamily value) noexcept {
  switch (value) {
  case hal::MemoryHandleFamily::invalid:
    return "invalid";
  case hal::MemoryHandleFamily::host_pointer:
    return "host_pointer";
  case hal::MemoryHandleFamily::cuda_device_pointer:
    return "cuda_device_pointer";
  case hal::MemoryHandleFamily::nvcv_image:
    return "nvcv_image";
  case hal::MemoryHandleFamily::ort_value:
    return "ort_value";
  case hal::MemoryHandleFamily::vulkan_buffer:
    return "vulkan_buffer";
  case hal::MemoryHandleFamily::vulkan_image:
    return "vulkan_image";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
OwnershipModelToken(hal::OwnershipModel value) noexcept {
  switch (value) {
  case hal::OwnershipModel::invalid:
    return "invalid";
  case hal::OwnershipModel::adapter_exclusive:
    return "adapter_exclusive";
  case hal::OwnershipModel::borrowed_until_completion:
    return "borrowed_until_completion";
  case hal::OwnershipModel::externally_owned:
    return "externally_owned";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
SynchronizationProtocolToken(hal::SynchronizationProtocol value) noexcept {
  switch (value) {
  case hal::SynchronizationProtocol::invalid:
    return "invalid";
  case hal::SynchronizationProtocol::host_synchronous:
    return "host_synchronous";
  case hal::SynchronizationProtocol::cuda_stream_ordered:
    return "cuda_stream_ordered";
  case hal::SynchronizationProtocol::maxine_bound_cuda_stream:
    return "maxine_bound_cuda_stream";
  case hal::SynchronizationProtocol::ort_provider_synchronous:
    return "ort_provider_synchronous";
  case hal::SynchronizationProtocol::vulkan_queue_fence:
    return "vulkan_queue_fence";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
PrimitiveToken(hal::PrimitiveOperationId value) noexcept {
  switch (value) {
  case hal::PrimitiveOperationId::invalid:
    return {};
  case hal::PrimitiveOperationId::convert_pixel_format:
    return "convert_pixel_format";
  case hal::PrimitiveOperationId::resize:
    return "resize";
  case hal::PrimitiveOperationId::mirror:
    return "mirror";
  case hal::PrimitiveOperationId::box_blur:
    return "box_blur";
  case hal::PrimitiveOperationId::composite:
    return "composite";
  case hal::PrimitiveOperationId::alpha_feather:
    return "alpha_feather";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
MultiIslandRouteToken(video::PreparedExecutionRouteId value) noexcept {
  switch (value) {
  case video::PreparedExecutionRouteId::invalid:
    return {};
  case video::PreparedExecutionRouteId::final_visual:
    return "final_visual";
  case video::PreparedExecutionRouteId::maxine_resident:
    return "maxine_resident";
  case video::PreparedExecutionRouteId::open_cuda_virtual_background:
    return "open_cuda_virtual_background";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
FailureActionToken(hal::FailureAction value) noexcept {
  switch (value) {
  case hal::FailureAction::fail_closed:
    return "fail_closed";
  case hal::FailureAction::fail_open_preserve_input:
    return "fail_open_preserve_input";
  case hal::FailureAction::fail_open_preserve_last_valid:
    return "fail_open_preserve_last_valid";
  }
  return {};
}

[[nodiscard]] constexpr std::string_view
CpuTailPlacementToken(hal::DeclaredCpuTail::Placement value) noexcept {
  switch (value) {
  case hal::DeclaredCpuTail::Placement::after_stage:
    return "after_stage";
  case hal::DeclaredCpuTail::Placement::before_stage:
    return "before_stage";
  }
  return {};
}

void AppendBackendId(std::ostream &out, hal::BackendId value) {
  out << "{\"id\":" << static_cast<std::uint32_t>(value) << ",\"token\":";
  AppendJsonString(out, BackendToken(value));
  out << '}';
}

void AppendProviderId(std::ostream &out, hal::SemanticEffectProviderId value) {
  out << "{\"id\":" << static_cast<std::uint32_t>(value) << ",\"token\":";
  AppendJsonString(out, ProviderToken(value));
  out << '}';
}

void AppendStatusId(std::ostream &out, hal::StatusId value) {
  out << "{\"id\":" << static_cast<std::uint32_t>(value) << ",\"token\":";
  AppendJsonString(out, StatusToken(value));
  out << '}';
}

void AppendReasonId(std::ostream &out, hal::ReasonId value) {
  out << "{\"id\":" << static_cast<std::uint32_t>(value) << ",\"token\":";
  AppendJsonString(out, ReasonToken(value));
  out << '}';
}

void AppendEffectId(std::ostream &out, hal::SemanticEffectId value) {
  out << "{\"id\":" << static_cast<std::uint32_t>(value) << ",\"token\":";
  AppendJsonString(out, hal::StableEffectId(value));
  out << '}';
}

void AppendEvidenceArray(
    std::ostream &out,
    const std::array<hal::CapabilityEvidenceRecord,
                     hal::kMaxCapabilityEvidenceRecords> &evidence,
    std::size_t count);

void AppendTypedEvidenceArray(
    std::ostream &out,
    const std::array<hal::CapabilityEvidenceRecord,
                     hal::kMaxCapabilityEvidenceRecords> &evidence,
    std::size_t count) {
  out << '[';
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0)
      out << ',';
    const auto &record = evidence[index];
    out << "{\"code\":{\"id\":" << static_cast<std::uint32_t>(record.code)
        << ",\"token\":";
    AppendJsonString(out, EvidenceToken(record.code));
    out << "},\"status\":";
    AppendStatusId(out, record.status);
    out << ",\"reason\":";
    AppendReasonId(out, record.reason);
    out << ",\"effect\":";
    AppendEffectId(out, record.semantic);
    out << ",\"backend\":";
    AppendBackendId(out, record.backend);
    out << ",\"satisfied\":" << (record.satisfied ? "true" : "false") << '}';
  }
  out << ']';
}

void AppendProviderCapability(std::ostream &out,
                              const hal::SemanticCapabilityEvidence &value) {
  out << "{\"effect\":";
  AppendEffectId(out, value.semantic);
  out << ",\"provider\":";
  AppendProviderId(out, value.provider);
  out << ",\"status\":";
  AppendStatusId(out, value.status);
  out << ",\"reason\":";
  AppendReasonId(out, value.reason);
  out << ",\"cpu_tail\":{\"id\":" << static_cast<std::uint32_t>(value.cpu_tail)
      << ",\"token\":";
  AppendJsonString(out, CpuTailToken(value.cpu_tail));
  out << "},\"cpu_tail_scratch_bytes\":" << value.cpu_tail_scratch_bytes
      << ",\"cpu_tail_placement\":{\"id\":"
      << static_cast<unsigned int>(value.cpu_tail_placement) << ",\"token\":";
  AppendJsonString(out, CpuTailPlacementToken(value.cpu_tail_placement));
  out << "},\"live_stage\":" << (value.live_stage ? "true" : "false")
      << ",\"production_eligible\":"
      << (value.production_eligible ? "true" : "false")
      << ",\"exact_evidence\":" << (value.exact_evidence ? "true" : "false")
      << ",\"supports_tracked_vignette_center\":"
      << (value.supports_tracked_vignette_center ? "true" : "false")
      << ",\"evidence\":";
  AppendTypedEvidenceArray(out, value.evidence, value.evidence_count);
  out << '}';
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

void AppendPreparedExecutionDomain(std::ostream &out,
                                   const hal::ExecutionDomainId &value);

void AppendWriterCommitIdentity(
    std::ostream &out, const video::WriterOutputCommitIdentity &value) {
  out << "{\"valid\":" << (value.valid ? "true" : "false")
      << ",\"capture_sequence\":" << value.capture_sequence
      << ",\"output_commit_sequence\":" << value.output_commit_sequence << '}';
}

void AppendMultiIslandResolution(std::ostream &out,
                                 const hal::ResolutionResult &value) {
  out << "{\"request_mode\":{\"id\":"
      << static_cast<unsigned int>(value.request_mode) << ",\"token\":";
  AppendJsonString(out, value.request_mode == hal::BackendRequestMode::automatic
                            ? std::string_view{"automatic"}
                            : std::string_view{"explicit_backend"});
  out << "},\"strict_backend_selection\":"
      << (value.strict_backend_selection ? "true" : "false") << ",\"status\":";
  AppendStatusId(out, value.status);
  out << ",\"reason\":";
  AppendReasonId(out, value.reason);
  out << ",\"requested_backend\":";
  AppendBackendId(out, value.requested_backend);
  out << ",\"selected_backend\":";
  AppendBackendId(out, value.selected_backend);
  out << ",\"active_backend\":";
  AppendBackendId(out, value.active_backend);
  out << ",\"selected_provider\":";
  AppendProviderId(out, value.selected_provider);
  out << ",\"available\":" << (value.available ? "true" : "false")
      << ",\"degraded\":" << (value.degraded ? "true" : "false")
      << ",\"fallback\":" << (value.fallback ? "true" : "false")
      << ",\"passthrough\":" << (value.passthrough ? "true" : "false")
      << ",\"stages\":[";
  for (std::size_t index = 0; index < value.stage_count; ++index) {
    if (index != 0)
      out << ',';
    const auto &stage = value.stages[index];
    out << "{\"stage_id\":" << stage.intent.stage_id << ",\"effect\":";
    AppendEffectId(out, stage.intent.semantic);
    out << ",\"required\":" << (stage.intent.required ? "true" : "false")
        << ",\"failure_action\":{\"id\":"
        << static_cast<unsigned int>(stage.intent.failure_action)
        << ",\"token\":";
    AppendJsonString(out, FailureActionToken(stage.intent.failure_action));
    out << "},\"requires_tracked_vignette_center\":"
        << (stage.intent.requires_tracked_vignette_center ? "true" : "false")
        << ",\"status\":";
    AppendStatusId(out, stage.status);
    out << ",\"reason\":";
    AppendReasonId(out, stage.reason);
    out << ",\"selected_backend\":";
    AppendBackendId(out, stage.selected_backend);
    out << ",\"provider\":";
    AppendProviderId(out, stage.provider);
    out << ",\"adapter_instance_id\":" << stage.adapter_instance_id
        << ",\"execution_domain\":";
    AppendPreparedExecutionDomain(out, stage.execution_domain);
    out << ",\"cpu_tail\":{\"id\":"
        << static_cast<std::uint32_t>(stage.cpu_tail) << ",\"token\":";
    AppendJsonString(out, CpuTailToken(stage.cpu_tail));
    out << "},\"cpu_tail_scratch_bytes\":" << stage.cpu_tail_scratch_bytes
        << ",\"cpu_tail_placement\":{\"id\":"
        << static_cast<unsigned int>(stage.cpu_tail_placement) << ",\"token\":";
    AppendJsonString(out, CpuTailPlacementToken(stage.cpu_tail_placement));
    out << "},\"skipped_optional\":"
        << (stage.skipped_optional ? "true" : "false")
        << ",\"fallback_selected\":"
        << (stage.fallback_selected ? "true" : "false") << ",\"evidence\":";
    AppendTypedEvidenceArray(out, stage.evidence, stage.evidence_count);
    out << '}';
  }
  out << "]}";
}

void AppendMultiIslandReadiness(std::ostream &out,
                                const video::TypedRuntimeReadiness &value) {
  out << "{\"configured\":" << (value.configured ? "true" : "false")
      << ",\"program_id\":" << value.program_id << ",\"invalidation_key\":";
  AppendInvalidationKey(out, value.invalidation_key);
  out << ",\"status\":";
  AppendStatusId(out, value.value.status);
  out << ",\"reason\":";
  AppendReasonId(out, value.value.reason);
  out << ",\"requested_backend\":";
  AppendBackendId(out, value.value.requested_backend);
  out << ",\"selected_backend\":";
  AppendBackendId(out, value.value.selected_backend);
  out << ",\"provider\":";
  AppendProviderId(out, value.value.provider);
  out << ",\"execution_domain\":";
  AppendPreparedExecutionDomain(out, value.value.execution_domain);
  out << ",\"ready_stage_mask\":" << value.value.ready_stage_mask
      << ",\"blocked_stage_mask\":" << value.value.blocked_stage_mask
      << ",\"disabled_stage_mask\":" << value.value.disabled_stage_mask
      << ",\"available\":" << (value.value.available ? "true" : "false")
      << ",\"degraded\":" << (value.value.degraded ? "true" : "false")
      << ",\"rebuild_requested\":"
      << (value.value.rebuild_requested ? "true" : "false") << ",\"evidence\":";
  AppendTypedEvidenceArray(out, value.value.evidence,
                           value.value.evidence_count);
  out << '}';
}

void AppendMultiIslandCommittedFrame(std::ostream &out,
                                     const hal::CommittedFrameSnapshot &value) {
  out << "{\"valid\":" << (value.valid ? "true" : "false")
      << ",\"program_id\":" << value.program_id << ",\"invalidation_key\":";
  AppendInvalidationKey(out, value.invalidation_key);
  out << ",\"execution_domain\":";
  AppendPreparedExecutionDomain(out, value.execution_domain);
  out << ",\"capture_sequence\":" << value.capture_sequence
      << ",\"output_commit_sequence\":" << value.output_commit_sequence
      << ",\"stages\":[";
  for (std::size_t index = 0; index < value.stage_count; ++index) {
    if (index != 0)
      out << ',';
    const auto &stage = value.stages[index];
    out << "{\"stage_id\":" << stage.stage_id << ",\"effect\":";
    AppendEffectId(out, stage.semantic);
    out << ",\"primitive\":{\"id\":"
        << static_cast<std::uint32_t>(stage.primitive) << ",\"token\":";
    AppendJsonString(out, PrimitiveToken(stage.primitive));
    out << "},\"backend\":";
    AppendBackendId(out, stage.backend);
    out << ",\"provider\":";
    AppendProviderId(out, stage.provider);
    out << ",\"status\":{\"id\":" << static_cast<unsigned int>(stage.status)
        << ",\"token\":";
    AppendJsonString(out, StageExecutionToken(stage.status));
    out << "},\"reason\":";
    AppendReasonId(out, stage.reason);
    out << '}';
  }
  out << "]}";
}

void AppendMultiIslandTelemetry(std::ostream &out,
                                const hal::ExecutionTelemetry &value,
                                const hal::ResolutionResult &resolution) {
  out << "{\"requested_backend\":";
  AppendBackendId(out, value.requested_backend);
  out << ",\"selected_backend\":";
  AppendBackendId(out, value.selected_backend);
  out << ",\"active_backend\":";
  AppendBackendId(out, value.active_backend);
  out << ",\"active_provider\":";
  AppendProviderId(out, value.active_provider);
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
  AppendStatusId(out, value.runtime_status);
  out << ",\"runtime_reason\":";
  AppendReasonId(out, value.runtime_reason);
  out << ",\"transfers\":";
  AppendTransferCounters(out, value.transfers);
  out << ",\"frame_latency\":";
  AppendLatencyHistogram(out, value.frame_latency);
  out << ",\"stage_latency\":[";
  for (std::size_t index = 0; index < resolution.stage_count; ++index) {
    if (index != 0)
      out << ',';
    out << "{\"stage_id\":" << resolution.stages[index].intent.stage_id
        << ",\"effect\":";
    AppendEffectId(out, resolution.stages[index].intent.semantic);
    out << ",\"latency\":";
    AppendLatencyHistogram(out, value.stage_latency[index]);
    out << ",\"committed_successes\":" << value.committed_stage_successes[index]
        << '}';
  }
  out << "]}";
}

void AppendPreparedIslandStageRecord(
    std::ostream &out, const video::PreparedIslandStageRecord &value) {
  out << "{\"stage_id\":" << value.stage_id << ",\"effect\":";
  AppendEffectId(out, value.effect);
  out << ",\"backend\":";
  AppendBackendId(out, value.backend);
  out << ",\"provider\":";
  AppendProviderId(out, value.provider);
  out << ",\"status\":";
  AppendStatusId(out, value.status);
  out << ",\"reason\":";
  AppendReasonId(out, value.reason);
  out << ",\"failure_action\":{\"id\":"
      << static_cast<unsigned int>(value.failure_action) << ",\"token\":";
  AppendJsonString(out, FailureActionToken(value.failure_action));
  out << "},\"committed_status\":{\"id\":"
      << static_cast<unsigned int>(value.committed_status) << ",\"token\":";
  AppendJsonString(out, StageExecutionToken(value.committed_status));
  out << "},\"required\":" << (value.required ? "true" : "false")
      << ",\"resolved\":" << (value.resolved ? "true" : "false")
      << ",\"committed\":" << (value.committed ? "true" : "false") << '}';
}

void AppendPreparedExecutionIsland(
    std::ostream &out, const video::PreparedExecutionIslandStatus &value) {
  out << "{\"route\":{\"id\":" << static_cast<std::uint32_t>(value.route)
      << ",\"token\":";
  AppendJsonString(out, MultiIslandRouteToken(value.route));
  out << "},\"island_id\":" << value.island_id
      << ",\"adapter_instance_id\":" << value.adapter_instance_id
      << ",\"snapshot_generation\":" << value.snapshot_generation
      << ",\"capability_generation\":" << value.capability_generation
      << ",\"backend\":";
  AppendBackendId(out, value.backend);
  out << ",\"execution_domain\":";
  AppendPreparedExecutionDomain(out, value.execution_domain);
  out << ",\"resolution\":";
  AppendMultiIslandResolution(out, value.resolution);
  out << ",\"readiness\":";
  AppendMultiIslandReadiness(out, value.readiness);
  out << ",\"telemetry\":";
  AppendMultiIslandTelemetry(out, value.telemetry, value.resolution);
  out << ",\"latest_committed_frame\":";
  AppendMultiIslandCommittedFrame(out, value.latest_committed_frame);
  out << ",\"stages\":[";
  for (std::size_t index = 0; index < value.stage_count; ++index) {
    if (index != 0)
      out << ',';
    AppendPreparedIslandStageRecord(out, value.stages[index]);
  }
  out << "],\"valid\":" << (value.valid ? "true" : "false")
      << ",\"current\":" << (value.current ? "true" : "false") << '}';
}

void AppendCapabilitySelection(
    std::ostream &out, video::ProductCapabilityLaneId lane,
    EffectCapabilityCode code, EffectCapabilityCode nested_code,
    const hal::SemanticCapabilityEvidence &capability) {
  out << "{\"lane\":{\"id\":" << static_cast<unsigned int>(lane)
      << ",\"token\":";
  AppendJsonString(out, LaneToken(lane));
  out << "},\"code\":";
  AppendCode(out, code);
  out << ",\"nested_code\":";
  AppendCode(out, nested_code);
  out << ",\"semantic\":";
  AppendProviderCapability(out, capability);
  out << '}';
}

void AppendStableEffectRuntimeState(
    std::ostream &out, const video::StableEffectRuntimeState &value) {
  out << "{\"effect\":";
  AppendEffectId(out, value.effect);
  out << ",\"requested_capability\":";
  AppendCapabilitySelection(out, value.capability_lane, value.capability_code,
                            value.nested_capability_code, value.capability);
  out << ",\"selected_capability\":";
  AppendCapabilitySelection(
      out, value.selected_capability_lane, value.selected_capability_code,
      value.selected_nested_capability_code, value.selected_capability);
  out << ",\"requested_backend\":";
  AppendBackendId(out, value.requested_backend);
  out << ",\"requested_provider\":";
  AppendProviderId(out, value.requested_provider);
  out << ",\"selected_backend\":";
  AppendBackendId(out, value.selected_backend);
  out << ",\"selected_provider\":";
  AppendProviderId(out, value.selected_provider);
  out << ",\"active_backend\":";
  AppendBackendId(out, value.active_backend);
  out << ",\"active_provider\":";
  AppendProviderId(out, value.active_provider);
  out << ",\"status\":";
  AppendStatusId(out, value.status);
  out << ",\"reason\":";
  AppendReasonId(out, value.reason);
  out << ",\"failure_action\":{\"id\":"
      << static_cast<unsigned int>(value.failure_action) << ",\"token\":";
  AppendJsonString(out, FailureActionToken(value.failure_action));
  out << "},\"commit\":";
  AppendWriterCommitIdentity(out, value.commit);
  out << ",\"selected_island_id\":" << value.selected_island_id
      << ",\"committed_stage_id\":" << value.committed_stage_id
      << ",\"committed_stage_status\":{\"id\":"
      << static_cast<unsigned int>(value.committed_stage_status)
      << ",\"token\":";
  AppendJsonString(out, StageExecutionToken(value.committed_stage_status));
  out << "},\"strict_backend_selection\":"
      << (value.strict_backend_selection ? "true" : "false")
      << ",\"strict_provider_selection\":"
      << (value.strict_provider_selection ? "true" : "false")
      << ",\"required\":" << (value.required ? "true" : "false")
      << ",\"requested\":" << (value.requested ? "true" : "false")
      << ",\"available\":" << (value.available ? "true" : "false")
      << ",\"selected\":" << (value.selected ? "true" : "false")
      << ",\"active\":" << (value.active ? "true" : "false")
      << ",\"degraded\":" << (value.degraded ? "true" : "false")
      << ",\"blocked\":" << (value.blocked ? "true" : "false")
      << ",\"fallback\":" << (value.fallback ? "true" : "false") << '}';
}

void AppendMultiIslandVideoRuntimeStatus(
    std::ostream &out,
    const video::MultiIslandVideoRuntimeStatusSnapshot &snapshot) {
  out << "{\"valid\":true,\"schema_version\":" << snapshot.schema_version
      << ",\"hal_contract_version\":" << snapshot.hal_contract_version
      << ",\"reason_code_version\":" << snapshot.reason_code_version
      << ",\"status_generation\":" << snapshot.status_generation
      << ",\"generations\":";
  AppendGenerations(out, snapshot.generations);
  out << ",\"program_generations\":";
  AppendProgramGenerations(out, snapshot.program_generations);
  out << ",\"lifecycle\":{\"id\":"
      << static_cast<unsigned int>(snapshot.lifecycle) << ",\"token\":";
  AppendJsonString(out, LifecycleToken(snapshot.lifecycle));
  out << "},\"capabilities\":";
  AppendCapabilitySnapshot(out, snapshot.capabilities);
  out << ",\"writer_commit\":";
  AppendWriterCommitIdentity(out, snapshot.writer_commit);
  out << ",\"islands\":[";
  for (std::size_t index = 0; index < snapshot.island_count; ++index) {
    if (index != 0)
      out << ',';
    AppendPreparedExecutionIsland(out, snapshot.islands[index]);
  }
  out << "],\"effects\":[";
  for (std::size_t index = 0; index < snapshot.effect_count; ++index) {
    if (index != 0)
      out << ',';
    AppendStableEffectRuntimeState(out, snapshot.effects[index]);
  }
  out << "],\"aggregate\":{\"current_island_count\":"
      << snapshot.aggregate.current_island_count << ",\"transfers\":";
  // The contract validator proves this aggregate equals the once-only sum of
  // current islands. Serialize the aggregate directly; never add island
  // counters again on the diagnostics path.
  AppendTransferCounters(out, snapshot.aggregate.transfers);
  out << "}}";
}

void AppendPreparedExecutionDomain(std::ostream &out,
                                   const hal::ExecutionDomainId &value) {
  out << "{\"backend\":{\"id\":" << static_cast<std::uint32_t>(value.backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(value.backend));
  out << "},\"api_family\":{\"id\":"
      << static_cast<std::uint32_t>(value.api_family) << ",\"token\":";
  AppendJsonString(out, ApiFamilyToken(value.api_family));
  out << "},\"physical_device\":{\"vendor_id\":"
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
      << value.logical_context_generation << ",\"allocator_family\":{\"id\":"
      << static_cast<std::uint32_t>(value.allocator_family) << ",\"token\":";
  AppendJsonString(out, AllocatorFamilyToken(value.allocator_family));
  out << "},\"memory_handle_family\":{\"id\":"
      << static_cast<std::uint32_t>(value.memory_handle_family)
      << ",\"token\":";
  AppendJsonString(out, MemoryHandleFamilyToken(value.memory_handle_family));
  out << "},\"queue_or_stream_identity\":" << value.queue_or_stream_identity
      << ",\"ownership_model\":{\"id\":"
      << static_cast<std::uint32_t>(value.ownership_model) << ",\"token\":";
  AppendJsonString(out, OwnershipModelToken(value.ownership_model));
  out << "},\"synchronization_protocol\":{\"id\":"
      << static_cast<std::uint32_t>(value.synchronization_protocol)
      << ",\"token\":";
  AppendJsonString(
      out, SynchronizationProtocolToken(value.synchronization_protocol));
  out << "},\"adapter_instance_id\":" << value.adapter_instance_id << '}';
}

[[nodiscard]] bool
HistogramIsWellFormed(const hal::LatencyHistogram &histogram) noexcept {
  std::uint64_t bucket_samples = 0;
  for (const std::uint64_t count : histogram.bucket_counts) {
    if (count > std::numeric_limits<std::uint64_t>::max() - bucket_samples)
      return false;
    bucket_samples += count;
  }
  if (bucket_samples != histogram.samples)
    return false;
  if (histogram.samples == 0) {
    return histogram.total_ns == 0 && histogram.maximum_ns == 0 &&
           histogram.p95_ns == 0 && histogram.p99_ns == 0;
  }
  return histogram.maximum_ns <= histogram.total_ns &&
         histogram.p95_ns <= histogram.p99_ns &&
         histogram.p99_ns <= histogram.maximum_ns;
}

struct PreparedSnapshotValidation {
  bool valid = true;
  std::string_view code = "valid";
};

[[nodiscard]] constexpr std::uint32_t
PreparedValidationId(std::string_view code) noexcept {
  if (code == "valid")
    return 0;
  if (code == "malformed_count")
    return 1;
  if (code == "inconsistent_telemetry")
    return 2;
  if (code == "inconsistent_live_setup")
    return 3;
  if (code == "wrong_execution_domain")
    return 4;
  if (code == "stale_committed_output")
    return 5;
  if (code == "malformed_committed_output")
    return 6;
  return std::numeric_limits<std::uint32_t>::max();
}

void FailPreparedValidation(PreparedSnapshotValidation *validation,
                            std::string_view code) noexcept {
  if (validation && validation->valid) {
    validation->valid = false;
    validation->code = code;
  }
}

PreparedSnapshotValidation ValidatePreparedSnapshot(
    const video::PreparedHardwareExecutionSnapshot &snapshot) noexcept {
  PreparedSnapshotValidation validation;
  const auto &setup = snapshot.setup;
  if (setup.resolution.stage_count > setup.resolution.stages.size() ||
      setup.domain_plan.island_count > setup.domain_plan.islands.size() ||
      setup.domain_plan.boundary_count > setup.domain_plan.boundaries.size() ||
      setup.shadow_comparison.discrepancy_count >
          setup.shadow_comparison.discrepancies.size() ||
      snapshot.readiness.evidence_count > snapshot.readiness.evidence.size() ||
      snapshot.latest_committed_frame.stage_count >
          snapshot.latest_committed_frame.stages.size()) {
    FailPreparedValidation(&validation, "malformed_count");
    return validation;
  }
  for (std::size_t index = 0; index < setup.resolution.stage_count; ++index) {
    if (setup.resolution.stages[index].evidence_count >
        setup.resolution.stages[index].evidence.size()) {
      FailPreparedValidation(&validation, "malformed_count");
      return validation;
    }
  }
  for (std::size_t index = 0; index < setup.domain_plan.island_count; ++index) {
    if (setup.domain_plan.islands[index].stage_count >
        setup.domain_plan.islands[index].resolved_stage_indices.size()) {
      FailPreparedValidation(&validation, "malformed_count");
      return validation;
    }
  }
  const auto &telemetry = snapshot.telemetry;
  if (telemetry.execution_successes > telemetry.execution_attempts ||
      telemetry.committed_output_frames > telemetry.execution_successes ||
      telemetry.active_backend_frames > telemetry.committed_output_frames ||
      telemetry.setup_successes > telemetry.setup_attempts ||
      !HistogramIsWellFormed(telemetry.frame_latency)) {
    FailPreparedValidation(&validation, "inconsistent_telemetry");
    return validation;
  }
  for (std::size_t index = 0; index < setup.resolution.stage_count; ++index) {
    if (!HistogramIsWellFormed(telemetry.stage_latency[index]) ||
        telemetry.committed_stage_successes[index] >
            telemetry.committed_output_frames) {
      FailPreparedValidation(&validation, "inconsistent_telemetry");
      return validation;
    }
  }

  if (snapshot.route != video::PreparedHardwareRoute::hal_live)
    return validation;
  if (snapshot.eligibility != video::PreparedHardwareEligibility::eligible ||
      !snapshot.prepared || !snapshot.active_program.valid ||
      snapshot.active_program.program_id == 0 ||
      snapshot.active_program.stage_count == 0 ||
      snapshot.active_program.stage_count >
          snapshot.setup.resolution.stages.size() ||
      setup.mode != video::HardwareExecutionSetupMode::hal_live ||
      setup.status != hal::StatusId::ready ||
      setup.resolution.status != hal::StatusId::ready ||
      !setup.resolution.available || setup.resolution.passthrough ||
      setup.resolution.stage_count == 0 ||
      setup.domain_plan.status != hal::StatusId::ready ||
      setup.domain_plan.passthrough || setup.domain_plan.island_count != 1 ||
      snapshot.readiness.status != hal::StatusId::ready ||
      !snapshot.readiness.available) {
    FailPreparedValidation(&validation, "inconsistent_live_setup");
    return validation;
  }
  const auto &island = setup.domain_plan.islands[0];
  if (!hal::ValidateExecutionDomain(island.execution_domain) ||
      snapshot.active_program.execution_domain != island.execution_domain ||
      snapshot.active_program.owner_adapter_instance_id !=
          island.adapter_instance_id ||
      snapshot.active_program.invalidation_key.canonical_plan_hash == 0 ||
      snapshot.active_program.invalidation_key.execution_domain_hash == 0 ||
      snapshot.active_program.invalidation_key.generations.context !=
          island.execution_domain.logical_context_generation ||
      snapshot.active_program.stage_count != setup.resolution.stage_count ||
      snapshot.readiness.execution_domain != island.execution_domain ||
      snapshot.readiness.selected_backend !=
          setup.resolution.selected_backend ||
      island.adapter_instance_id !=
          island.execution_domain.adapter_instance_id ||
      island.stage_count != setup.resolution.stage_count) {
    FailPreparedValidation(&validation, "wrong_execution_domain");
    return validation;
  }
  for (std::size_t index = 0; index < setup.resolution.stage_count; ++index) {
    const auto &stage = setup.resolution.stages[index];
    if (stage.execution_domain != island.execution_domain ||
        stage.adapter_instance_id != island.adapter_instance_id ||
        stage.selected_backend != setup.resolution.selected_backend) {
      FailPreparedValidation(&validation, "wrong_execution_domain");
      return validation;
    }
  }
  const auto &commit = snapshot.latest_committed_frame;
  if (!commit.valid)
    return validation;
  if (commit.program_id != snapshot.active_program.program_id ||
      commit.invalidation_key != snapshot.active_program.invalidation_key) {
    FailPreparedValidation(&validation, "stale_committed_output");
    return validation;
  }
  if (commit.execution_domain != snapshot.active_program.execution_domain ||
      commit.execution_domain != snapshot.readiness.execution_domain) {
    FailPreparedValidation(&validation, "wrong_execution_domain");
    return validation;
  }
  if (commit.stage_count != snapshot.active_program.stage_count ||
      commit.stage_count != setup.resolution.stage_count) {
    FailPreparedValidation(&validation, "malformed_committed_output");
    return validation;
  }
  for (std::size_t index = 0; index < commit.stage_count; ++index) {
    const auto &activity = commit.stages[index];
    const auto &resolved = setup.resolution.stages[index];
    if (activity.stage_id != resolved.intent.stage_id ||
        activity.semantic != resolved.intent.semantic ||
        activity.backend != resolved.selected_backend ||
        activity.provider != resolved.provider) {
      FailPreparedValidation(&validation, "malformed_committed_output");
      return validation;
    }
  }
  return validation;
}

[[nodiscard]] bool CommitMatchesPreparedSetup(
    const video::PreparedHardwareExecutionSnapshot &snapshot) noexcept {
  const auto &commit = snapshot.latest_committed_frame;
  if (!commit.valid || commit.program_id == 0 || commit.capture_sequence == 0 ||
      commit.output_commit_sequence == 0 || commit.stage_count == 0 ||
      !snapshot.active_program.valid ||
      commit.program_id != snapshot.active_program.program_id ||
      commit.invalidation_key != snapshot.active_program.invalidation_key ||
      commit.execution_domain != snapshot.active_program.execution_domain ||
      commit.stage_count != snapshot.active_program.stage_count ||
      commit.execution_domain != snapshot.readiness.execution_domain ||
      commit.invalidation_key.canonical_plan_hash == 0 ||
      commit.invalidation_key.execution_domain_hash == 0 ||
      commit.invalidation_key.generations.configuration == 0 ||
      commit.invalidation_key.generations.geometry == 0 ||
      commit.invalidation_key.generations.context !=
          commit.execution_domain.logical_context_generation ||
      commit.stage_count != snapshot.setup.resolution.stage_count) {
    return false;
  }
  for (std::size_t index = 0; index < commit.stage_count; ++index) {
    const auto &activity = commit.stages[index];
    const auto &resolved = snapshot.setup.resolution.stages[index];
    if (activity.stage_id != resolved.intent.stage_id ||
        activity.semantic != resolved.intent.semantic ||
        activity.backend != resolved.selected_backend ||
        activity.provider != resolved.provider) {
      return false;
    }
  }
  return true;
}

void ClampPreparedCounts(
    video::PreparedHardwareExecutionSnapshot *snapshot) noexcept {
  if (!snapshot)
    return;
  auto &setup = snapshot->setup;
  setup.resolution.stage_count =
      std::min(setup.resolution.stage_count, setup.resolution.stages.size());
  setup.domain_plan.island_count = std::min(setup.domain_plan.island_count,
                                            setup.domain_plan.islands.size());
  setup.domain_plan.boundary_count = std::min(
      setup.domain_plan.boundary_count, setup.domain_plan.boundaries.size());
  setup.shadow_comparison.discrepancy_count =
      std::min(setup.shadow_comparison.discrepancy_count,
               setup.shadow_comparison.discrepancies.size());
  snapshot->readiness.evidence_count = std::min(
      snapshot->readiness.evidence_count, snapshot->readiness.evidence.size());
  snapshot->latest_committed_frame.stage_count =
      std::min(snapshot->latest_committed_frame.stage_count,
               snapshot->latest_committed_frame.stages.size());
  for (std::size_t index = 0; index < setup.resolution.stage_count; ++index) {
    auto &stage = setup.resolution.stages[index];
    stage.evidence_count =
        std::min(stage.evidence_count, stage.evidence.size());
  }
  for (std::size_t index = 0; index < setup.domain_plan.island_count; ++index) {
    auto &island = setup.domain_plan.islands[index];
    island.stage_count =
        std::min(island.stage_count, island.resolved_stage_indices.size());
  }
}

void AppendPreparedReadiness(std::ostream &out,
                             const hal::ReadinessSnapshot &readiness) {
  out << "{\"status\":{\"id\":" << static_cast<std::uint32_t>(readiness.status)
      << ",\"token\":";
  AppendJsonString(out, StatusToken(readiness.status));
  out << "},\"reason\":{\"id\":" << static_cast<std::uint32_t>(readiness.reason)
      << ",\"token\":";
  AppendJsonString(out, ReasonToken(readiness.reason));
  out << "},\"requested_backend\":{\"id\":"
      << static_cast<std::uint32_t>(readiness.requested_backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(readiness.requested_backend));
  out << "},\"selected_backend\":{\"id\":"
      << static_cast<std::uint32_t>(readiness.selected_backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(readiness.selected_backend));
  out << "},\"provider\":{\"id\":"
      << static_cast<std::uint32_t>(readiness.provider) << ",\"token\":";
  AppendJsonString(out, ProviderToken(readiness.provider));
  out << "},\"execution_domain\":";
  AppendPreparedExecutionDomain(out, readiness.execution_domain);
  out << ",\"ready_stage_mask\":" << readiness.ready_stage_mask
      << ",\"blocked_stage_mask\":" << readiness.blocked_stage_mask
      << ",\"disabled_stage_mask\":" << readiness.disabled_stage_mask
      << ",\"available\":" << (readiness.available ? "true" : "false")
      << ",\"degraded\":" << (readiness.degraded ? "true" : "false")
      << ",\"rebuild_requested\":"
      << (readiness.rebuild_requested ? "true" : "false") << ",\"evidence\":";
  AppendEvidenceArray(out, readiness.evidence, readiness.evidence_count);
  out << '}';
}

void AppendPreparedDomainPlan(std::ostream &out, const hal::DomainPlan &plan) {
  out << "{\"status\":{\"id\":" << static_cast<std::uint32_t>(plan.status)
      << ",\"token\":";
  AppendJsonString(out, StatusToken(plan.status));
  out << "},\"reason\":{\"id\":" << static_cast<std::uint32_t>(plan.reason)
      << ",\"token\":";
  AppendJsonString(out, ReasonToken(plan.reason));
  out << "},\"passthrough\":" << (plan.passthrough ? "true" : "false")
      << ",\"expected_transfers\":";
  AppendTransferCounters(out, plan.expected_transfers);
  out << ",\"islands\":[";
  for (std::size_t index = 0; index < plan.island_count; ++index) {
    if (index != 0)
      out << ',';
    const auto &island = plan.islands[index];
    out << "{\"adapter_instance_id\":" << island.adapter_instance_id
        << ",\"execution_domain\":";
    AppendPreparedExecutionDomain(out, island.execution_domain);
    out << ",\"stage_count\":" << island.stage_count << ",\"requires_upload\":"
        << (island.requires_upload ? "true" : "false")
        << ",\"requires_final_readback\":"
        << (island.requires_final_readback ? "true" : "false") << '}';
  }
  out << "],\"boundaries\":[";
  for (std::size_t index = 0; index < plan.boundary_count; ++index) {
    if (index != 0)
      out << ',';
    const auto &boundary = plan.boundaries[index];
    out << "{\"kind\":{\"id\":" << static_cast<unsigned int>(boundary.kind)
        << ",\"token\":";
    AppendJsonString(out, boundary.kind ==
                                  hal::ExecutionBoundaryKind::adapter_transfer
                              ? std::string_view{"adapter_transfer"}
                              : std::string_view{"cpu_tail"});
    out << "},\"after_stage_id\":" << boundary.after_stage_id
        << ",\"before_stage_id\":" << boundary.before_stage_id
        << ",\"boundary_index\":" << boundary.boundary_index
        << ",\"cpu_tail\":";
    AppendJsonString(out, CpuTailToken(boundary.cpu_tail));
    out << ",\"scratch_bytes\":" << boundary.scratch_bytes
        << ",\"readback_required\":"
        << (boundary.readback_required ? "true" : "false")
        << ",\"following_upload_required\":"
        << (boundary.following_upload_required ? "true" : "false")
        << ",\"source_domain\":";
    AppendPreparedExecutionDomain(out, boundary.source_domain);
    out << ",\"destination_domain\":";
    AppendPreparedExecutionDomain(out, boundary.destination_domain);
    out << '}';
  }
  out << "]}";
}

void AppendPreparedCommittedFrame(std::ostream &out,
                                  const hal::CommittedFrameSnapshot &commit,
                                  bool current_program_match,
                                  bool expose_active) {
  out << "{\"valid\":" << (commit.valid ? "true" : "false")
      << ",\"current_program_match\":"
      << (current_program_match ? "true" : "false")
      << ",\"active_attribution_valid\":" << (expose_active ? "true" : "false")
      << ",\"program_id\":" << commit.program_id << ",\"invalidation_key\":";
  AppendInvalidationKey(out, commit.invalidation_key);
  out << ",\"execution_domain\":";
  AppendPreparedExecutionDomain(out, commit.execution_domain);
  out << ",\"capture_sequence\":" << commit.capture_sequence
      << ",\"output_commit_sequence\":" << commit.output_commit_sequence
      << ",\"stages\":[";
  for (std::size_t index = 0; index < commit.stage_count; ++index) {
    if (index != 0)
      out << ',';
    const auto &stage = commit.stages[index];
    out << "{\"stage_id\":" << stage.stage_id
        << ",\"effect\":{\"id\":" << static_cast<std::uint32_t>(stage.semantic)
        << ",\"token\":";
    AppendJsonString(out, hal::StableEffectId(stage.semantic));
    out << "},\"primitive\":{\"id\":"
        << static_cast<std::uint32_t>(stage.primitive) << ",\"token\":";
    AppendJsonString(out, PrimitiveToken(stage.primitive));
    out << "},\"backend\":{\"id\":" << static_cast<std::uint32_t>(stage.backend)
        << ",\"token\":";
    AppendJsonString(out, BackendToken(stage.backend));
    out << "},\"provider\":{\"id\":"
        << static_cast<std::uint32_t>(stage.provider) << ",\"token\":";
    AppendJsonString(out, ProviderToken(stage.provider));
    out << "},\"status\":{\"id\":" << static_cast<unsigned int>(stage.status)
        << ",\"token\":";
    AppendJsonString(out, StageExecutionToken(stage.status));
    out << "},\"reason\":{\"id\":" << static_cast<std::uint32_t>(stage.reason)
        << ",\"token\":";
    AppendJsonString(out, ReasonToken(stage.reason));
    out << "}}";
  }
  out << "]}";
}

void AppendPreparedTelemetry(std::ostream &out,
                             const hal::ExecutionTelemetry &telemetry,
                             const hal::ResolutionResult &resolution) {
  out << "{\"requested_backend\":{\"id\":"
      << static_cast<std::uint32_t>(telemetry.requested_backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(telemetry.requested_backend));
  out << "},\"selected_backend\":{\"id\":"
      << static_cast<std::uint32_t>(telemetry.selected_backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(telemetry.selected_backend));
  out << "},\"cumulative_active_backend\":{\"id\":"
      << static_cast<std::uint32_t>(telemetry.active_backend) << ",\"token\":";
  AppendJsonString(out, BackendToken(telemetry.active_backend));
  out << "},\"cumulative_active_provider\":{\"id\":"
      << static_cast<std::uint32_t>(telemetry.active_provider) << ",\"token\":";
  AppendJsonString(out, ProviderToken(telemetry.active_provider));
  out << "},\"execution_attempts\":" << telemetry.execution_attempts
      << ",\"execution_successes\":" << telemetry.execution_successes
      << ",\"committed_output_frames\":" << telemetry.committed_output_frames
      << ",\"active_backend_frames\":" << telemetry.active_backend_frames
      << ",\"fallback_frames\":" << telemetry.fallback_frames
      << ",\"optional_fail_open_stages\":"
      << telemetry.optional_fail_open_stages
      << ",\"required_stage_failures\":" << telemetry.required_stage_failures
      << ",\"temporal_resets\":" << telemetry.temporal_resets
      << ",\"device_losses\":" << telemetry.device_losses
      << ",\"setup_attempts\":" << telemetry.setup_attempts
      << ",\"setup_successes\":" << telemetry.setup_successes
      << ",\"setup_rollbacks\":" << telemetry.setup_rollbacks
      << ",\"setup_retention_rejections\":"
      << telemetry.setup_retention_rejections
      << ",\"circuit_breaker_opens\":" << telemetry.circuit_breaker_opens
      << ",\"disabled_stage_frames\":" << telemetry.disabled_stage_frames
      << ",\"rebuild_requests\":" << telemetry.rebuild_requests
      << ",\"frame_drop_events\":" << telemetry.frame_drop_events
      << ",\"dropped_capture_frames\":" << telemetry.dropped_capture_frames
      << ",\"preserve_order_failures\":" << telemetry.preserve_order_failures
      << ",\"latest_frame_wins_events\":" << telemetry.latest_frame_wins_events
      << ",\"disabled_stage_mask\":" << telemetry.disabled_stage_mask
      << ",\"runtime_status\":{\"id\":"
      << static_cast<std::uint32_t>(telemetry.runtime_status) << ",\"token\":";
  AppendJsonString(out, StatusToken(telemetry.runtime_status));
  out << "},\"runtime_reason\":{\"id\":"
      << static_cast<std::uint32_t>(telemetry.runtime_reason) << ",\"token\":";
  AppendJsonString(out, ReasonToken(telemetry.runtime_reason));
  out << "},\"transfers\":";
  AppendTransferCounters(out, telemetry.transfers);
  out << ",\"frame_latency\":";
  AppendLatencyHistogram(out, telemetry.frame_latency);
  out << ",\"stage_timing\":[";
  for (std::size_t index = 0; index < resolution.stage_count; ++index) {
    if (index != 0)
      out << ',';
    out << "{\"stage_id\":" << resolution.stages[index].intent.stage_id
        << ",\"effect\":";
    AppendJsonString(
        out, hal::StableEffectId(resolution.stages[index].intent.semantic));
    out << ",\"latency\":";
    AppendLatencyHistogram(out, telemetry.stage_latency[index]);
    out << ",\"committed_successes\":"
        << telemetry.committed_stage_successes[index] << '}';
  }
  out << "]}";
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

MultiIslandVideoRuntimeStatusJsonResult
SerializeMultiIslandVideoRuntimeStatusJson(
    const video::MultiIslandVideoRuntimeStatusSnapshot &snapshot) {
  MultiIslandVideoRuntimeStatusJsonResult result;
  result.validation =
      video::ValidateMultiIslandVideoRuntimeStatusSnapshot(snapshot);
  if (result.validation.code != video::MultiIslandStatusValidationCode::valid) {
    return result;
  }
  std::ostringstream out;
  AppendMultiIslandVideoRuntimeStatus(out, snapshot);
  result.json = out.str();
  result.ok = true;
  return result;
}

PreparedHardwareExecutionJsonResult SerializePreparedHardwareExecutionJson(
    const video::PreparedHardwareExecutionSnapshot &snapshot,
    bool supervisor_pipeline_running, bool supervisor_pipeline_starting) {
  PreparedHardwareExecutionJsonResult result;
  const PreparedSnapshotValidation validation =
      ValidatePreparedSnapshot(snapshot);
  auto safe = snapshot;
  ClampPreparedCounts(&safe);

  const bool live_runtime_status =
      safe.telemetry.runtime_status != hal::StatusId::quarantined &&
      safe.telemetry.runtime_status != hal::StatusId::stale &&
      safe.telemetry.runtime_status != hal::StatusId::stopped &&
      safe.telemetry.runtime_status != hal::StatusId::failed &&
      safe.telemetry.runtime_reason != hal::ReasonId::device_lost &&
      safe.telemetry.runtime_reason != hal::ReasonId::shutdown_requested;
  const bool exact_program_match =
      validation.valid && CommitMatchesPreparedSetup(safe);
  const bool current_commit =
      exact_program_match && supervisor_pipeline_running &&
      !supervisor_pipeline_starting &&
      safe.route == video::PreparedHardwareRoute::hal_live &&
      safe.eligibility == video::PreparedHardwareEligibility::eligible &&
      safe.prepared && live_runtime_status;
  bool successful_effect_committed = false;
  if (current_commit) {
    for (std::size_t index = 0; index < safe.latest_committed_frame.stage_count;
         ++index) {
      if (safe.latest_committed_frame.stages[index].status ==
          hal::StageExecutionStatus::succeeded) {
        successful_effect_committed = true;
        break;
      }
    }
  }
  const bool expose_active = current_commit && successful_effect_committed;

  std::ostringstream out;
  out << "{\"valid\":" << (validation.valid ? "true" : "false")
      << ",\"schema_version\":1,\"hal_contract_version\":"
      << hal::kContractVersion
      << ",\"validation\":{\"id\":" << PreparedValidationId(validation.code)
      << ",\"code\":";
  AppendJsonString(out, validation.code);
  out << "},\"route\":{\"id\":" << static_cast<unsigned int>(safe.route)
      << ",\"token\":";
  AppendJsonString(out, PreparedRouteToken(safe.route));
  out << "},\"eligibility\":{\"id\":"
      << static_cast<unsigned int>(safe.eligibility) << ",\"token\":";
  AppendJsonString(out, PreparedEligibilityToken(safe.eligibility));
  out << "},\"prepared\":" << (safe.prepared ? "true" : "false")
      << ",\"supervisor\":{\"pipeline_running\":"
      << (supervisor_pipeline_running ? "true" : "false")
      << ",\"pipeline_starting\":"
      << (supervisor_pipeline_starting ? "true" : "false")
      << "},\"setup\":{\"mode\":{\"id\":"
      << static_cast<unsigned int>(safe.setup.mode) << ",\"token\":";
  AppendJsonString(out, SetupModeToken(safe.setup.mode));
  out << "},\"status\":{\"id\":"
      << static_cast<std::uint32_t>(safe.setup.status) << ",\"token\":";
  AppendJsonString(out, StatusToken(safe.setup.status));
  out << "},\"reason\":{\"id\":"
      << static_cast<std::uint32_t>(safe.setup.reason) << ",\"token\":";
  AppendJsonString(out, ReasonToken(safe.setup.reason));
  out << "},\"legacy_path_retained_for_setup_rollback\":"
      << (safe.setup.legacy_path_retained_for_setup_rollback ? "true" : "false")
      << ",\"shadow_compared\":"
      << (safe.setup.shadow_compared ? "true" : "false");
  if (safe.setup.shadow_compared) {
    out << ",\"shadow_comparison\":{\"exact_match\":"
        << (safe.setup.shadow_comparison.exact_match ? "true" : "false")
        << ",\"truncated\":"
        << (safe.setup.shadow_comparison.truncated ? "true" : "false")
        << ",\"discrepancy_count\":"
        << safe.setup.shadow_comparison.discrepancy_count << '}';
  }
  out << ",\"resolution\":";
  AppendResolution(out, safe.setup.resolution);
  out << ",\"domain_plan\":";
  AppendPreparedDomainPlan(out, safe.setup.domain_plan);
  out << "},\"active_program\":{\"valid\":"
      << (safe.active_program.valid ? "true" : "false")
      << ",\"program_id\":" << safe.active_program.program_id
      << ",\"owner_adapter_instance_id\":"
      << safe.active_program.owner_adapter_instance_id
      << ",\"stage_count\":" << safe.active_program.stage_count
      << ",\"invalidation_key\":";
  AppendInvalidationKey(out, safe.active_program.invalidation_key);
  out << ",\"execution_domain\":";
  AppendPreparedExecutionDomain(out, safe.active_program.execution_domain);
  out << "},\"readiness\":";
  AppendPreparedReadiness(out, safe.readiness);
  out << ",\"attribution\":{\"requested_backend\":{\"id\":"
      << static_cast<std::uint32_t>(safe.setup.resolution.requested_backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(safe.setup.resolution.requested_backend));
  out << "},\"selected_backend\":{\"id\":"
      << static_cast<std::uint32_t>(safe.setup.resolution.selected_backend)
      << ",\"token\":";
  AppendJsonString(out, BackendToken(safe.setup.resolution.selected_backend));
  out << "},\"active_backend\":{\"id\":";
  hal::BackendId active_backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId active_provider =
      hal::SemanticEffectProviderId::invalid;
  bool provider_initialized = false;
  bool mixed_providers = false;
  if (expose_active) {
    for (std::size_t index = 0; index < safe.latest_committed_frame.stage_count;
         ++index) {
      const auto &stage = safe.latest_committed_frame.stages[index];
      if (stage.status != hal::StageExecutionStatus::succeeded)
        continue;
      active_backend = stage.backend;
      if (!provider_initialized) {
        active_provider = stage.provider;
        provider_initialized = true;
      } else if (active_provider != stage.provider) {
        mixed_providers = true;
      }
    }
  }
  if (mixed_providers)
    active_provider = hal::SemanticEffectProviderId::invalid;
  out << static_cast<std::uint32_t>(active_backend) << ",\"token\":";
  AppendJsonString(out, BackendToken(active_backend));
  out << "},\"active_provider\":{\"id\":"
      << static_cast<std::uint32_t>(active_provider) << ",\"token\":";
  AppendJsonString(out, ProviderToken(active_provider));
  out << "},\"committed_stages\":[";
  bool first_active_stage = true;
  if (expose_active) {
    for (std::size_t index = 0; index < safe.latest_committed_frame.stage_count;
         ++index) {
      const auto &stage = safe.latest_committed_frame.stages[index];
      if (stage.status != hal::StageExecutionStatus::succeeded)
        continue;
      if (!first_active_stage)
        out << ',';
      first_active_stage = false;
      out << "{\"stage_id\":" << stage.stage_id << ",\"effect\":{\"id\":"
          << static_cast<std::uint32_t>(stage.semantic) << ",\"token\":";
      AppendJsonString(out, hal::StableEffectId(stage.semantic));
      out << "},\"backend\":{\"id\":"
          << static_cast<std::uint32_t>(stage.backend) << ",\"token\":";
      AppendJsonString(out, BackendToken(stage.backend));
      out << "},\"provider\":{\"id\":"
          << static_cast<std::uint32_t>(stage.provider) << ",\"token\":";
      AppendJsonString(out, ProviderToken(stage.provider));
      out << "}}";
    }
  }
  out << "]},\"telemetry\":";
  AppendPreparedTelemetry(out, safe.telemetry, safe.setup.resolution);
  out << ",\"latest_committed_frame\":";
  AppendPreparedCommittedFrame(out, safe.latest_committed_frame,
                               exact_program_match, expose_active);
  out << '}';

  result.ok = true;
  result.snapshot_valid = validation.valid;
  result.active_attribution_valid = expose_active;
  result.validation_code = validation.code;
  result.json = out.str();
  return result;
}

} // namespace studiocast::daemon
