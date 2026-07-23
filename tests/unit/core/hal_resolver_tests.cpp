#include "core/hal/domain_planner.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string_view>

using namespace studiocast::hal;

namespace {

std::atomic<bool> g_count_allocations{false};
std::atomic<std::uint64_t> g_allocations{0};

bool Require(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

ExecutionDomainId Domain(BackendId backend, std::uint64_t adapter,
                         std::uint8_t stable_byte) {
  ExecutionDomainId domain;
  domain.backend = backend;
  domain.api_family =
      backend == BackendId::vulkan
          ? ApiFamily::vulkan
          : (backend == BackendId::maxine
                 ? ApiFamily::maxine_nvcv
                 : (backend == BackendId::cpu
                        ? ApiFamily::host
                        : (backend == BackendId::open_video_onnxruntime
                               ? ApiFamily::onnxruntime
                               : ApiFamily::cuda_driver)));
  domain.physical_device.vendor_id =
      backend == BackendId::vulkan ? 0x1002u : 0x10deu;
  domain.physical_device.device_id = stable_byte;
  domain.physical_device.stable_id[0] = stable_byte;
  domain.logical_context_generation = 7 + adapter;
  domain.allocator_family =
      backend == BackendId::vulkan
          ? AllocatorFamily::vulkan
          : (backend == BackendId::maxine
                 ? AllocatorFamily::maxine_nvcv
                 : (backend == BackendId::cpu
                        ? AllocatorFamily::host
                        : (backend == BackendId::open_video_onnxruntime
                               ? AllocatorFamily::onnxruntime
                               : AllocatorFamily::cuda_driver)));
  domain.memory_handle_family =
      backend == BackendId::vulkan
          ? MemoryHandleFamily::vulkan_buffer
          : (backend == BackendId::maxine
                 ? MemoryHandleFamily::nvcv_image
                 : (backend == BackendId::cpu
                        ? MemoryHandleFamily::host_pointer
                        : (backend == BackendId::open_video_onnxruntime
                               ? MemoryHandleFamily::ort_value
                               : MemoryHandleFamily::cuda_device_pointer)));
  domain.queue_or_stream_identity = 100 + adapter;
  domain.ownership_model = OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      backend == BackendId::vulkan
          ? SynchronizationProtocol::vulkan_queue_fence
          : (backend == BackendId::maxine
                 ? SynchronizationProtocol::maxine_bound_cuda_stream
                 : (backend == BackendId::cpu
                        ? SynchronizationProtocol::host_synchronous
                        : (backend == BackendId::open_video_onnxruntime
                               ? SynchronizationProtocol::
                                     ort_provider_synchronous
                               : SynchronizationProtocol::
                                     cuda_stream_ordered)));
  domain.adapter_instance_id = adapter;
  return domain;
}

class FakeAdapter final : public IResidentAdapter {
public:
  FakeAdapter(std::uint64_t instance, BackendId backend)
      : instance_(instance), backend_(backend) {}

  std::uint64_t adapter_instance_id() const noexcept override {
    return instance_;
  }
  BackendId backend_id() const noexcept override { return backend_; }
  StatusId Prepare(const PreparedProgram &, PreparedProgramHandle *,
                   ReadinessSnapshot *, ReasonId *) noexcept override {
    return StatusId::blocked;
  }
  ExecuteResult Execute(const PreparedProgramHandle &,
                        const ExecuteRequest &) noexcept override {
    return {};
  }
  StatusId ResetTemporalState(const PreparedProgramHandle &,
                              ReasonId) noexcept override {
    return StatusId::ready;
  }
  void DestroyPrepared(PreparedProgramHandle *) noexcept override {}
  void Shutdown() noexcept override {}

private:
  std::uint64_t instance_;
  BackendId backend_;
};

SemanticCapabilityEvidence
Capability(SemanticEffectId semantic, SemanticEffectProviderId provider,
           StatusId status = StatusId::ready, ReasonId reason = ReasonId::none,
           bool production = true, CpuTailId cpu_tail = CpuTailId::invalid) {
  SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.provider = provider;
  capability.status = status;
  capability.reason = reason;
  capability.cpu_tail = cpu_tail;
  capability.cpu_tail_scratch_bytes = cpu_tail == CpuTailId::invalid ? 0 : 4096;
  capability.live_stage = production;
  capability.production_eligible = production;
  capability.exact_evidence = production;
  return capability;
}

AdapterRegistration Registration(FakeAdapter *adapter, BackendId backend,
                                 std::uint32_t priority,
                                 std::uint8_t stable_byte) {
  AdapterRegistration registration;
  registration.adapter_instance_id = adapter->adapter_instance_id();
  registration.backend = backend;
  registration.execution_domain =
      Domain(backend, adapter->adapter_instance_id(), stable_byte);
  registration.selection_priority = priority;
  registration.resident_adapter = adapter;
  return registration;
}

bool Register(StaticAdapterRegistry *registry,
              const AdapterRegistration &registration) {
  ReasonId reason = ReasonId::none;
  return Require(registry->Register(registration, &reason) == StatusId::ready,
                 "valid registration was rejected");
}

CanonicalStageIntent Stage(std::uint32_t id, SemanticEffectId semantic,
                           bool required = true) {
  CanonicalStageIntent stage;
  stage.stage_id = id;
  stage.semantic = semantic;
  stage.required = required;
  stage.failure_action = required
                             ? FailureAction::fail_closed
                             : FailureAction::fail_open_preserve_last_valid;
  return stage;
}

ResolutionRequest AutoRequest() {
  ResolutionRequest request;
  request.backend.mode = BackendRequestMode::automatic;
  request.backend.backend = BackendId::invalid;
  request.backend.strict = false;
  return request;
}

ResolutionRequest StrictRequest(BackendId backend) {
  ResolutionRequest request;
  request.backend.mode = BackendRequestMode::explicit_backend;
  request.backend.backend = backend;
  request.backend.strict = true;
  return request;
}

bool TestPrimitiveAndSemanticEvidenceRemainSeparate() {
  FakeAdapter cpu(10, BackendId::cpu);
  auto registration = Registration(&cpu, BackendId::cpu, 400, 10);
  registration.primitive_capabilities[0] = {
      PrimitiveOperationId::mirror, StatusId::ready, ReasonId::none, true};
  registration.primitive_capability_count = 1;

  StaticAdapterRegistry registry;
  bool ok = Register(&registry, registration);
  const auto *primitive =
      registry.FindPrimitive(10, PrimitiveOperationId::mirror);
  ok &= Require(primitive != nullptr && primitive->available,
                "registered CPU primitive was not discoverable");

  auto request = StrictRequest(BackendId::cpu);
  request.stages[0] = Stage(1, SemanticEffectId::mirror);
  request.stage_count = 1;
  const auto result = ResolveSemanticStages(registry, request);
  ok &= Require(result.status == StatusId::blocked &&
                    result.reason == ReasonId::semantic_effect_unsupported &&
                    result.selected_backend == BackendId::invalid,
                "CPU primitive incorrectly promoted CPU effect capability");

  FakeAdapter maxine(11, BackendId::maxine);
  auto maxine_registration = Registration(&maxine, BackendId::maxine, 100, 11);
  maxine_registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::vignette,
                 SemanticEffectProviderId::maxine_cuda_utility);
  maxine_registration.semantic_capability_count = 1;
  StaticAdapterRegistry maxine_registry;
  ok &= Register(&maxine_registry, maxine_registration);
  auto vignette = StrictRequest(BackendId::maxine);
  vignette.stages[0] = Stage(1, SemanticEffectId::vignette);
  vignette.stage_count = 1;
  const auto vignette_result = ResolveSemanticStages(maxine_registry, vignette);
  ok &= Require(vignette_result.status == StatusId::ready &&
                    vignette_result.selected_provider ==
                        SemanticEffectProviderId::maxine_cuda_utility,
                "Maxine vignette lost exact CUDA utility provider attribution");
  return ok;
}

bool TestDeterministicPrecedenceAndStrictSelection() {
  FakeAdapter cuda(20, BackendId::cuda);
  FakeAdapter maxine(21, BackendId::maxine);
  auto cuda_registration = Registration(&cuda, BackendId::cuda, 200, 20);
  cuda_registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::mirror,
                 SemanticEffectProviderId::open_video_onnxruntime_cuda);
  cuda_registration.semantic_capability_count = 1;
  auto maxine_registration = Registration(&maxine, BackendId::maxine, 100, 21);
  maxine_registration.semantic_capabilities[0] = Capability(
      SemanticEffectId::mirror, SemanticEffectProviderId::maxine_vfx);
  maxine_registration.semantic_capability_count = 1;

  StaticAdapterRegistry first;
  StaticAdapterRegistry reverse;
  bool ok = Register(&first, cuda_registration);
  ok &= Register(&first, maxine_registration);
  ok &= Register(&reverse, maxine_registration);
  ok &= Register(&reverse, cuda_registration);

  auto automatic = AutoRequest();
  automatic.stages[0] = Stage(1, SemanticEffectId::mirror);
  automatic.stage_count = 1;
  const auto first_result = ResolveSemanticStages(first, automatic);
  const auto reverse_result = ResolveSemanticStages(reverse, automatic);
  ok &= Require(first_result.selected_backend == BackendId::maxine &&
                    reverse_result.selected_backend == BackendId::maxine &&
                    first_result.stages[0].adapter_instance_id == 21 &&
                    reverse_result.stages[0].adapter_instance_id == 21,
                "selection precedence depended on registration order");

  auto strict_cuda = StrictRequest(BackendId::cuda);
  strict_cuda.stages[0] = Stage(1, SemanticEffectId::mirror);
  strict_cuda.stage_count = 1;
  const auto cuda_result = ResolveSemanticStages(first, strict_cuda);
  ok &= Require(cuda_result.status == StatusId::ready &&
                    cuda_result.requested_backend == BackendId::cuda &&
                    cuda_result.selected_backend == BackendId::cuda,
                "strict CUDA request did not select exact CUDA backend");

  auto strict_vulkan = StrictRequest(BackendId::vulkan);
  strict_vulkan.stages[0] = Stage(1, SemanticEffectId::mirror);
  strict_vulkan.stage_count = 1;
  const auto blocked = ResolveSemanticStages(first, strict_vulkan);
  ok &= Require(blocked.status == StatusId::blocked &&
                    blocked.reason == ReasonId::strict_backend_unavailable,
                "strict request silently selected another GPU backend");

  StaticAdapterRegistry fallback_registry;
  ok &= Register(&fallback_registry, maxine_registration);
  auto unavailable_cuda = cuda_registration;
  unavailable_cuda.semantic_capabilities[0] = Capability(
      SemanticEffectId::mirror,
      SemanticEffectProviderId::open_video_onnxruntime_cuda, StatusId::blocked,
      ReasonId::capability_evidence_incomplete, false);
  ok &= Register(&fallback_registry, unavailable_cuda);
  auto preferred_cuda = StrictRequest(BackendId::cuda);
  preferred_cuda.backend.strict = false;
  preferred_cuda.stages[0] = Stage(1, SemanticEffectId::mirror);
  preferred_cuda.stage_count = 1;
  const auto fallback =
      ResolveSemanticStages(fallback_registry, preferred_cuda);
  ok &= Require(
      fallback.status == StatusId::degraded && fallback.degraded &&
          fallback.fallback &&
          fallback.reason == ReasonId::capability_evidence_incomplete &&
          fallback.requested_backend == BackendId::cuda &&
          fallback.selected_backend == BackendId::maxine &&
          fallback.active_backend == BackendId::invalid &&
          fallback.stages[0].fallback_selected &&
          fallback.stages[0].status == StatusId::degraded &&
          fallback.stages[0].reason == ReasonId::capability_evidence_incomplete,
      "non-strict backend fallback lost exact attribution");

  StaticAdapterRegistry no_requested_evidence_registry;
  ok &= Register(&no_requested_evidence_registry, maxine_registration);
  const auto provider_unavailable =
      ResolveSemanticStages(no_requested_evidence_registry, preferred_cuda);
  ok &= Require(
      provider_unavailable.status == StatusId::degraded &&
          provider_unavailable.fallback &&
          provider_unavailable.reason ==
              ReasonId::semantic_provider_unavailable,
      "fallback without requested-backend evidence lost its typed reason");

  auto live_fallback = fallback;
  live_fallback.fallback = false;
  live_fallback.stages[0].fallback_selected = false;
  const auto fallback_comparison =
      CompareShadowResolution(live_fallback, fallback);
  ok &= Require(
      !fallback_comparison.exact_match &&
          fallback_comparison.discrepancy_count == 2 &&
          fallback_comparison.discrepancies[0].field == ShadowField::fallback &&
          fallback_comparison.discrepancies[1].field ==
              ShadowField::stage_fallback,
      "shadow comparison did not expose fallback attribution discrepancy");
  return ok;
}

bool TestExplicitSemanticCpuProviderDoesNotLeakIntoStrictGpuSelection() {
  // This fake record exercises the extension contract only. StudioCast does
  // not register a production CPU semantic provider by default.
  FakeAdapter open_video(25, BackendId::open_video_onnxruntime);
  auto registration =
      Registration(&open_video, BackendId::open_video_onnxruntime, 350, 25);
  registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::virtual_background_blur,
                 SemanticEffectProviderId::open_video_onnxruntime_cpu);
  registration.semantic_capability_count = 1;
  StaticAdapterRegistry registry;
  bool ok = Register(&registry, registration);

  auto automatic = AutoRequest();
  automatic.stages[0] = Stage(1, SemanticEffectId::virtual_background_blur);
  automatic.stage_count = 1;
  const auto automatic_result = ResolveSemanticStages(registry, automatic);
  ok &= Require(automatic_result.status == StatusId::ready &&
                    automatic_result.selected_backend ==
                        BackendId::open_video_onnxruntime &&
                    automatic_result.selected_provider ==
                        SemanticEffectProviderId::open_video_onnxruntime_cpu,
                "explicit semantic Open Video CPU evidence was not selectable");
  const auto cpu_plan =
      PlanExecutionDomains(registry, automatic_result,
                           CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(cpu_plan.status == StatusId::ready &&
                    ClassifyExecutionResidency(registration.execution_domain) ==
                        ExecutionResidencyClass::host &&
                    cpu_plan.island_count == 1 &&
                    !cpu_plan.islands[0].requires_upload &&
                    !cpu_plan.islands[0].requires_final_readback &&
                    cpu_plan.expected_transfers.uploads == 0 &&
                    cpu_plan.expected_transfers.final_readbacks == 0 &&
                    cpu_plan.expected_transfers.intermediate_uploads == 0 &&
                    cpu_plan.expected_transfers.intermediate_readbacks == 0 &&
                    cpu_plan.expected_transfers.synchronizations == 0,
                "CPU semantic provider introduced GPU transfers");

  FakeAdapter vulkan(26, BackendId::vulkan);
  auto vulkan_registration = Registration(&vulkan, BackendId::vulkan, 300, 26);
  vulkan_registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::mirror,
                 SemanticEffectProviderId::open_vulkan_production_kernels);
  vulkan_registration.semantic_capability_count = 1;
  ok &= Register(&registry, vulkan_registration);
  auto mixed_request = automatic;
  mixed_request.stages[1] = Stage(2, SemanticEffectId::mirror);
  mixed_request.stage_count = 2;
  const auto mixed_resolution = ResolveSemanticStages(registry, mixed_request);
  const auto mixed_plan =
      PlanExecutionDomains(registry, mixed_resolution,
                           CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(
      mixed_plan.status == StatusId::ready && mixed_plan.island_count == 2 &&
          mixed_plan.boundary_count == 1 &&
          !mixed_plan.boundaries[0].readback_required &&
          mixed_plan.boundaries[0].following_upload_required &&
          mixed_plan.expected_transfers.uploads == 0 &&
          mixed_plan.expected_transfers.intermediate_readbacks == 0 &&
          mixed_plan.expected_transfers.intermediate_uploads == 1 &&
          mixed_plan.expected_transfers.final_readbacks == 1 &&
          mixed_plan.expected_transfers.synchronizations == 1,
      "host-to-device split transfer or synchronization accounting changed");

  auto strict_open_video = StrictRequest(BackendId::open_video_onnxruntime);
  strict_open_video.stages[0] =
      Stage(1, SemanticEffectId::virtual_background_blur);
  strict_open_video.stage_count = 1;
  const auto exact = ResolveSemanticStages(registry, strict_open_video);
  ok &= Require(exact.status == StatusId::ready &&
                    exact.selected_backend == BackendId::open_video_onnxruntime,
                "strict Open Video request lost its exact CPU provider");

  for (const BackendId backend :
       {BackendId::cuda, BackendId::vulkan, BackendId::maxine}) {
    auto strict_gpu = StrictRequest(backend);
    strict_gpu.stages[0] = Stage(1, SemanticEffectId::virtual_background_blur);
    strict_gpu.stage_count = 1;
    const auto blocked = ResolveSemanticStages(registry, strict_gpu);
    ok &= Require(blocked.status == StatusId::blocked &&
                      blocked.reason == ReasonId::strict_backend_unavailable &&
                      blocked.selected_backend == BackendId::invalid,
                  "strict GPU request selected the Open Video CPU provider");
  }
  return ok;
}

bool TestGoldenIntentOptionalStageAndShadowComparison() {
  FakeAdapter maxine(30, BackendId::maxine);
  auto registration = Registration(&maxine, BackendId::maxine, 100, 30);
  registration.semantic_capabilities[0] = Capability(
      SemanticEffectId::mirror, SemanticEffectProviderId::maxine_vfx);
  registration.semantic_capability_count = 1;
  StaticAdapterRegistry registry;
  bool ok = Register(&registry, registration);

  auto request = AutoRequest();
  request.stages[0] = Stage(10, SemanticEffectId::mirror);
  request.stages[1] = Stage(20, SemanticEffectId::eye_contact, false);
  request.stage_count = 2;
  const auto shadow = ResolveSemanticStages(registry, request);

  // Golden setup-time status fixture. Active remains invalid until execution
  // succeeds and the writer commits the resulting output.
  ok &= Require(shadow.requested_backend == BackendId::invalid &&
                    shadow.selected_backend == BackendId::maxine &&
                    shadow.active_backend == BackendId::invalid &&
                    shadow.status == StatusId::degraded &&
                    shadow.reason == ReasonId::semantic_effect_unsupported &&
                    shadow.stages[1].skipped_optional &&
                    shadow.stages[1].status == StatusId::degraded,
                "requested/selected/active optional-stage golden changed");

  auto live = shadow;
  live.selected_backend = BackendId::cuda;
  live.active_backend = BackendId::cuda;
  live.stages[0].selected_backend = BackendId::cuda;
  const auto comparison = CompareShadowResolution(live, shadow);
  ok &= Require(
      !comparison.exact_match && comparison.discrepancy_count == 3 &&
          comparison.discrepancies[0].field == ShadowField::selected_backend &&
          comparison.discrepancies[1].field == ShadowField::active_backend &&
          comparison.discrepancies[2].field == ShadowField::stage_backend,
      "shadow comparison did not emit exact stable discrepancies");
  ok &= Require(live.selected_backend == BackendId::cuda &&
                    live.active_backend == BackendId::cuda,
                "shadow comparison changed the live decision");
  return ok;
}

bool TestCrossAdapterPlanningAndCpuTails() {
  FakeAdapter cuda(40, BackendId::cuda);
  FakeAdapter vulkan(41, BackendId::vulkan);
  auto cuda_registration = Registration(&cuda, BackendId::cuda, 200, 40);
  cuda_registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::eye_contact,
                 SemanticEffectProviderId::open_video_onnxruntime_cuda);
  cuda_registration.semantic_capability_count = 1;
  auto vulkan_registration = Registration(&vulkan, BackendId::vulkan, 300, 41);
  vulkan_registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::mirror,
                 SemanticEffectProviderId::open_vulkan_production_kernels);
  vulkan_registration.semantic_capability_count = 1;

  StaticAdapterRegistry registry;
  bool ok = Register(&registry, cuda_registration);
  ok &= Register(&registry, vulkan_registration);
  auto request = AutoRequest();
  request.stages[0] = Stage(1, SemanticEffectId::eye_contact);
  request.stages[1] = Stage(2, SemanticEffectId::mirror);
  request.stage_count = 2;
  const auto resolution = ResolveSemanticStages(registry, request);
  ok &= Require(resolution.status == StatusId::ready &&
                    resolution.selected_backend == BackendId::invalid,
                "mixed exact providers did not retain per-stage attribution");

  const auto rejected =
      PlanExecutionDomains(registry, resolution, CrossAdapterPolicy::reject);
  ok &= Require(rejected.status == StatusId::blocked &&
                    rejected.reason ==
                        ReasonId::cross_adapter_residency_forbidden,
                "cross-adapter resident island was not rejected");
  const auto split = PlanExecutionDomains(
      registry, resolution, CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(split.status == StatusId::ready && split.island_count == 2 &&
                    split.boundary_count == 1 &&
                    split.boundaries[0].kind ==
                        ExecutionBoundaryKind::adapter_transfer &&
                    split.expected_transfers.uploads == 1 &&
                    split.expected_transfers.final_readbacks == 1 &&
                    split.expected_transfers.intermediate_uploads == 1 &&
                    split.expected_transfers.intermediate_readbacks == 1 &&
                    split.expected_transfers.synchronizations == 2,
                "cross-adapter split did not expose measured transfer islands");

  auto foreign_resolution = resolution;
  foreign_resolution.stages[0].adapter_instance_id = 9999;
  const auto foreign =
      PlanExecutionDomains(registry, foreign_resolution,
                           CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(foreign.status == StatusId::blocked &&
                    foreign.reason == ReasonId::foreign_adapter,
                "foreign resolved adapter was accepted by the planner");

  auto wrong_domain_resolution = resolution;
  ++wrong_domain_resolution.stages[0]
        .execution_domain.logical_context_generation;
  const auto wrong_domain =
      PlanExecutionDomains(registry, wrong_domain_resolution,
                           CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(wrong_domain.status == StatusId::blocked &&
                    wrong_domain.reason == ReasonId::wrong_execution_domain,
                "stale resolved execution domain was accepted by the planner");

  FakeAdapter maxine(42, BackendId::maxine);
  auto maxine_registration = Registration(&maxine, BackendId::maxine, 100, 42);
  maxine_registration.semantic_capabilities[0] = Capability(
      SemanticEffectId::auto_frame, SemanticEffectProviderId::maxine_ar,
      StatusId::degraded, ReasonId::cpu_tail_required, true,
      CpuTailId::auto_frame_tracking);
  maxine_registration.semantic_capabilities[1] = Capability(
      SemanticEffectId::mirror, SemanticEffectProviderId::maxine_vfx);
  maxine_registration.semantic_capability_count = 2;
  StaticAdapterRegistry tail_registry;
  ok &= Register(&tail_registry, maxine_registration);
  auto tail_request = AutoRequest();
  tail_request.stages[0] = Stage(1, SemanticEffectId::auto_frame);
  tail_request.stages[1] = Stage(2, SemanticEffectId::mirror);
  tail_request.stage_count = 2;
  const auto tail_resolution =
      ResolveSemanticStages(tail_registry, tail_request);
  const auto tail_plan =
      PlanExecutionDomains(tail_registry, tail_resolution,
                           CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(
      tail_plan.status == StatusId::degraded && tail_plan.island_count == 2 &&
          tail_plan.boundary_count == 1 &&
          tail_plan.boundaries[0].kind == ExecutionBoundaryKind::cpu_tail &&
          tail_plan.boundaries[0].after_stage_id == 1 &&
          tail_plan.boundaries[0].before_stage_id == 2 &&
          tail_plan.expected_transfers.cpu_tail_boundaries == 1 &&
          tail_plan.expected_transfers.uploads == 1 &&
          tail_plan.expected_transfers.final_readbacks == 1 &&
          tail_plan.expected_transfers.intermediate_readbacks == 1 &&
          tail_plan.expected_transfers.intermediate_uploads == 1 &&
          tail_plan.expected_transfers.synchronizations == 2,
      "CPU tail did not split and expose the resident boundary");

  tail_request.stage_count = 1;
  const auto final_tail_resolution =
      ResolveSemanticStages(tail_registry, tail_request);
  const auto final_tail_plan =
      PlanExecutionDomains(tail_registry, final_tail_resolution,
                           CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(
      final_tail_plan.island_count == 1 &&
          final_tail_plan.boundary_count == 1 &&
          final_tail_plan.expected_transfers.uploads == 1 &&
          final_tail_plan.expected_transfers.final_readbacks == 1 &&
          final_tail_plan.expected_transfers.intermediate_readbacks == 0 &&
          final_tail_plan.expected_transfers.intermediate_uploads == 0 &&
          final_tail_plan.expected_transfers.cpu_tail_boundaries == 1 &&
          final_tail_plan.expected_transfers.synchronizations == 1,
      "final CPU tail was incorrectly counted as a resumed island");
  return ok;
}

bool TestVulkanEligibilityPassthroughAndCapacity() {
  FakeAdapter vulkan(50, BackendId::vulkan);
  auto registration = Registration(&vulkan, BackendId::vulkan, 300, 50);
  registration.semantic_capabilities[0] =
      Capability(SemanticEffectId::mirror,
                 SemanticEffectProviderId::open_vulkan_production_kernels);
  registration.semantic_capabilities[1] =
      Capability(SemanticEffectId::vignette,
                 SemanticEffectProviderId::open_vulkan_production_kernels);
  registration.semantic_capabilities[2] = Capability(
      SemanticEffectId::virtual_background_blur,
      SemanticEffectProviderId::open_vulkan_production_kernels,
      StatusId::blocked, ReasonId::capability_evidence_incomplete, false);
  registration.semantic_capability_count = 3;
  StaticAdapterRegistry registry;
  bool ok = Register(&registry, registration);

  auto request = StrictRequest(BackendId::vulkan);
  request.stages[0] = Stage(1, SemanticEffectId::vignette);
  request.stages[1] = Stage(2, SemanticEffectId::mirror);
  request.stages[2] =
      Stage(3, SemanticEffectId::virtual_background_blur, false);
  request.stage_count = 3;
  const auto result = ResolveSemanticStages(registry, request);
  ok &= Require(result.status == StatusId::degraded &&
                    result.selected_backend == BackendId::vulkan &&
                    !result.stages[0].skipped_optional &&
                    !result.stages[1].skipped_optional &&
                    result.stages[2].skipped_optional &&
                    result.stages[2].reason ==
                        ReasonId::capability_evidence_incomplete,
                "Vulkan eligibility was not limited to vignette and mirror");

  auto invalid = Registration(&vulkan, BackendId::vulkan, 300, 51);
  invalid.adapter_instance_id = 51;
  invalid.execution_domain = Domain(BackendId::vulkan, 51, 51);
  invalid.resident_adapter = nullptr;
  invalid.semantic_capabilities[0] =
      Capability(SemanticEffectId::virtual_background_blur,
                 SemanticEffectProviderId::open_vulkan_production_kernels);
  invalid.semantic_capability_count = 1;
  ReasonId reason = ReasonId::none;
  StaticAdapterRegistry invalid_registry;
  ok &= Require(invalid_registry.Register(invalid, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::malformed_contract,
                "unqualified Vulkan semantic registered as production");

  auto incoherent_domain = registration;
  incoherent_domain.execution_domain.allocator_family =
      AllocatorFamily::cuda_driver;
  StaticAdapterRegistry incoherent_registry;
  ok &= Require(
      ClassifyExecutionResidency(incoherent_domain.execution_domain) ==
              ExecutionResidencyClass::invalid &&
          incoherent_registry.Register(incoherent_domain, &reason) ==
              StatusId::blocked &&
          reason == ReasonId::malformed_contract,
      "incoherent execution-domain family evidence did not fail closed");

  auto wrong_provider = registration;
  wrong_provider.adapter_instance_id = 53;
  wrong_provider.execution_domain = Domain(BackendId::vulkan, 53, 53);
  wrong_provider.resident_adapter = nullptr;
  wrong_provider.semantic_capabilities[0].provider =
      SemanticEffectProviderId::maxine_vfx;
  wrong_provider.semantic_capabilities[0].production_eligible = false;
  wrong_provider.semantic_capabilities[0].live_stage = false;
  wrong_provider.semantic_capabilities[0].exact_evidence = false;
  wrong_provider.semantic_capability_count = 1;
  StaticAdapterRegistry wrong_provider_registry;
  ok &= Require(wrong_provider_registry.Register(wrong_provider, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::malformed_contract,
                "backend/provider mismatch did not fail closed");

  auto missing_adapter = registration;
  missing_adapter.adapter_instance_id = 52;
  missing_adapter.execution_domain = Domain(BackendId::vulkan, 52, 52);
  missing_adapter.resident_adapter = nullptr;
  StaticAdapterRegistry missing_adapter_registry;
  ok &= Require(missing_adapter_registry.Register(missing_adapter, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::malformed_contract,
                "live production semantic accepted a null resident adapter");

  auto tracked_vignette = StrictRequest(BackendId::vulkan);
  tracked_vignette.stages[0] = Stage(1, SemanticEffectId::vignette);
  tracked_vignette.stages[0].requires_tracked_vignette_center = true;
  tracked_vignette.stage_count = 1;
  const auto tracked_result = ResolveSemanticStages(registry, tracked_vignette);
  ok &= Require(tracked_result.status == StatusId::blocked &&
                    tracked_result.reason ==
                        ReasonId::semantic_provider_unavailable,
                "Vulkan tracked-center vignette was promoted to production");

  const auto passthrough =
      ResolveSemanticStages(registry, StrictRequest(BackendId::vulkan));
  const auto passthrough_plan = PlanExecutionDomains(
      registry, passthrough, CrossAdapterPolicy::split_with_measured_transfer);
  ok &= Require(
      passthrough.status == StatusId::passthrough &&
          passthrough_plan.passthrough && passthrough_plan.island_count == 0 &&
          passthrough_plan.expected_transfers.uploads == 0 &&
          passthrough_plan.expected_transfers.final_readbacks == 0 &&
          passthrough_plan.expected_transfers.intermediate_readbacks == 0 &&
          passthrough_plan.expected_transfers.intermediate_uploads == 0,
      "zero-effect passthrough created GPU work");

  StaticAdapterRegistry full;
  for (std::size_t index = 0; index < kMaxRegisteredAdapters; ++index) {
    FakeAdapter *unused_adapter = nullptr;
    AdapterRegistration entry;
    entry.adapter_instance_id = 1000 + index;
    entry.backend = BackendId::test_fake;
    entry.execution_domain =
        Domain(BackendId::test_fake, entry.adapter_instance_id,
               static_cast<std::uint8_t>(index + 1));
    entry.selection_priority = static_cast<std::uint32_t>(index + 1);
    entry.resident_adapter = unused_adapter;
    ok &= Register(&full, entry);
  }
  AdapterRegistration overflow;
  overflow.adapter_instance_id = 2000;
  overflow.backend = BackendId::test_fake;
  overflow.execution_domain = Domain(BackendId::test_fake, 2000, 77);
  overflow.selection_priority = 99;
  ok &= Require(full.Register(overflow, &reason) ==
                        StatusId::resource_exhausted &&
                    reason == ReasonId::resource_exhausted,
                "fixed registry capacity did not fail closed");
  return ok;
}

bool TestNoAllocationInResolvePlanAndShadow() {
  FakeAdapter maxine(60, BackendId::maxine);
  auto registration = Registration(&maxine, BackendId::maxine, 100, 60);
  registration.semantic_capabilities[0] = Capability(
      SemanticEffectId::mirror, SemanticEffectProviderId::maxine_vfx);
  registration.semantic_capability_count = 1;
  StaticAdapterRegistry registry;
  bool ok = Register(&registry, registration);
  auto request = AutoRequest();
  request.stages[0] = Stage(1, SemanticEffectId::mirror);
  request.stage_count = 1;

  g_allocations.store(0);
  g_count_allocations.store(true);
  const auto resolution = ResolveSemanticStages(registry, request);
  const auto plan = PlanExecutionDomains(
      registry, resolution, CrossAdapterPolicy::split_with_measured_transfer);
  const auto comparison = CompareShadowResolution(resolution, resolution);
  g_count_allocations.store(false);
  ok &= Require(g_allocations.load() == 0,
                "resolver/planner/shadow comparison allocated");
  ok &= Require(plan.status == StatusId::ready && plan.island_count == 1 &&
                    plan.expected_transfers.uploads == 1 &&
                    plan.expected_transfers.final_readbacks == 1 &&
                    plan.expected_transfers.intermediate_readbacks == 0 &&
                    plan.expected_transfers.intermediate_uploads == 0 &&
                    plan.expected_transfers.synchronizations == 1 &&
                    comparison.exact_match,
                "allocation test setup did not resolve");
  return ok;
}

} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

int main() {
  bool ok = true;
  ok &= TestPrimitiveAndSemanticEvidenceRemainSeparate();
  ok &= TestDeterministicPrecedenceAndStrictSelection();
  ok &= TestExplicitSemanticCpuProviderDoesNotLeakIntoStrictGpuSelection();
  ok &= TestGoldenIntentOptionalStageAndShadowComparison();
  ok &= TestCrossAdapterPlanningAndCpuTails();
  ok &= TestVulkanEligibilityPassthroughAndCapacity();
  ok &= TestNoAllocationInResolvePlanAndShadow();
  if (ok)
    std::cout << "StudioCast HAL resolver tests passed\n";
  return ok ? 0 : 1;
}
