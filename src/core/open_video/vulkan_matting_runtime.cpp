#include "core/open_video/vulkan_matting_runtime.h"

#include <cctype>
#include <limits>
#include <sstream>
#include <utility>

namespace studiocast::open_vulkan {

namespace {

bool CheckedMul(std::size_t a, std::size_t b, std::size_t *out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  *out = a * b;
  return true;
}

bool ExpectedF32Bytes(int n, int c, int h, int w, std::size_t *bytes_out) {
  if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
    return false;
  std::size_t elements = 1;
  if (!CheckedMul(elements, static_cast<std::size_t>(n), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(c), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(h), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(w), &elements) ||
      !CheckedMul(elements, sizeof(float), bytes_out)) {
    return false;
  }
  return true;
}

bool IsSha256Hex(const std::string &value) {
  if (value.size() != 64)
    return false;
  for (const char c : value) {
    if (!std::isxdigit(static_cast<unsigned char>(c)))
      return false;
  }
  return true;
}

bool CompleteGraphDescriptor(const VulkanMattingGraphDescriptor &graph) {
  return !graph.param_path.empty() && !graph.bin_path.empty() &&
         IsSha256Hex(graph.param_sha256) && IsSha256Hex(graph.bin_sha256) &&
         !graph.input_blob.empty() && !graph.output_blob.empty() &&
         !graph.converter_name.empty() && !graph.converter_version.empty() &&
         (graph.precision == "fp32" || graph.precision == "fp16");
}

bool SameDevice(const VulkanMattingDeviceContext &expected,
                const VulkanMattingDeviceContext &actual) {
  if (!expected.ownership_domain ||
      expected.ownership_domain != actual.ownership_domain ||
      expected.physical_device == 0 ||
      expected.physical_device != actual.physical_device ||
      expected.logical_device == 0 ||
      expected.logical_device != actual.logical_device || expected.queue == 0 ||
      expected.queue != actual.queue ||
      expected.queue_family_index != actual.queue_family_index ||
      expected.vendor_id != actual.vendor_id ||
      expected.device_id != actual.device_id || expected.context_id == 0 ||
      expected.context_id != actual.context_id ||
      expected.context_generation == 0 ||
      expected.context_generation != actual.context_generation ||
      !actual.non_cpu_device_selected || !actual.compute_queue_available ||
      !actual.context_healthy) {
    return false;
  }
  return expected.stable_device_id.empty() ||
         expected.stable_device_id == actual.stable_device_id;
}

std::string PrefixError(const char *operation, const std::string &detail) {
  std::string out = "Open Vulkan matting ";
  out += operation;
  out += " failed";
  if (!detail.empty()) {
    out += ": ";
    out += detail;
  }
  return out;
}

class UnavailableVulkanMattingRuntime final : public VulkanMattingRuntime {
public:
  explicit UnavailableVulkanMattingRuntime(std::string reason)
      : reason_(std::move(reason)) {
    evidence_.runtime_name = "unavailable";
  }

  VulkanMattingRuntimeResult
  Initialize(const VulkanMattingDeviceContext &) override {
    return VulkanMattingRuntimeResult::Failure(
        VulkanMattingRuntimeFailure::unavailable, reason_);
  }
  VulkanMattingRuntimeResult
  LoadGraph(const VulkanMattingGraphDescriptor &) override {
    return VulkanMattingRuntimeResult::Failure(
        VulkanMattingRuntimeFailure::unavailable, reason_);
  }
  VulkanMattingRuntimeResult PreparePersistentResources(
      const VulkanMattingPersistentResources &) override {
    return VulkanMattingRuntimeResult::Failure(
        VulkanMattingRuntimeFailure::unavailable, reason_);
  }
  VulkanMattingRuntimeResult Warmup() override {
    return VulkanMattingRuntimeResult::Failure(
        VulkanMattingRuntimeFailure::unavailable, reason_);
  }
  VulkanMattingRuntimeResult Run(const VulkanMattingBufferBinding &,
                                 const VulkanMattingBufferBinding &) override {
    return VulkanMattingRuntimeResult::Failure(
        VulkanMattingRuntimeFailure::unavailable, reason_);
  }
  VulkanMattingRuntimeEvidence Evidence() const override { return evidence_; }

private:
  std::string reason_;
  VulkanMattingRuntimeEvidence evidence_;
};

} // namespace

const char *
VulkanMattingRuntimeFailureName(VulkanMattingRuntimeFailure failure) {
  switch (failure) {
  case VulkanMattingRuntimeFailure::none:
    return "none";
  case VulkanMattingRuntimeFailure::unavailable:
    return "unavailable";
  case VulkanMattingRuntimeFailure::out_of_memory:
    return "out_of_memory";
  case VulkanMattingRuntimeFailure::device_lost:
    return "device_lost";
  case VulkanMattingRuntimeFailure::invalid_graph:
    return "invalid_graph";
  case VulkanMattingRuntimeFailure::invalid_shape:
    return "invalid_shape";
  case VulkanMattingRuntimeFailure::cpu_layer_rejected:
    return "cpu_layer_rejected";
  case VulkanMattingRuntimeFailure::device_identity_mismatch:
    return "device_identity_mismatch";
  case VulkanMattingRuntimeFailure::context_generation_mismatch:
    return "context_generation_mismatch";
  case VulkanMattingRuntimeFailure::residency_check_failed:
    return "residency_check_failed";
  case VulkanMattingRuntimeFailure::cpu_readback_rejected:
    return "cpu_readback_rejected";
  case VulkanMattingRuntimeFailure::synchronization_failed:
    return "synchronization_failed";
  case VulkanMattingRuntimeFailure::allocation_contract_failed:
    return "allocation_contract_failed";
  case VulkanMattingRuntimeFailure::execution_failed:
    return "execution_failed";
  }
  return "unknown";
}

namespace {

std::string FailureBlocker(const VulkanMattingReadinessInput &input) {
  switch (input.failure) {
  case VulkanMattingRuntimeFailure::device_lost:
    return kOpenVulkanMattingDeviceLostReason;
  case VulkanMattingRuntimeFailure::device_identity_mismatch:
  case VulkanMattingRuntimeFailure::context_generation_mismatch:
    return kOpenVulkanMattingDeviceMismatchReason;
  case VulkanMattingRuntimeFailure::cpu_layer_rejected:
    return kOpenVulkanMattingCpuLayerReason;
  case VulkanMattingRuntimeFailure::cpu_readback_rejected:
    return kOpenVulkanMattingCpuReadbackReason;
  case VulkanMattingRuntimeFailure::residency_check_failed:
    return kOpenVulkanMattingResidencyReason;
  case VulkanMattingRuntimeFailure::synchronization_failed:
    return kOpenVulkanMattingSynchronizationReason;
  case VulkanMattingRuntimeFailure::allocation_contract_failed:
  case VulkanMattingRuntimeFailure::out_of_memory:
    return kOpenVulkanMattingAllocationReason;
  case VulkanMattingRuntimeFailure::unavailable:
    return kOpenVulkanMattingAdapterUnavailableReason;
  case VulkanMattingRuntimeFailure::invalid_graph:
  case VulkanMattingRuntimeFailure::invalid_shape:
  case VulkanMattingRuntimeFailure::execution_failed:
    return !input.session_warmed ? kOpenVulkanMattingWarmupReason
                                 : kOpenVulkanMattingRuntimeFailureReason;
  case VulkanMattingRuntimeFailure::none:
    break;
  }
  return kOpenVulkanMattingRuntimeFailureReason;
}

VulkanMattingReadiness Blocked(const VulkanMattingReadinessInput &input,
                               std::string blocker, std::string detail) {
  VulkanMattingReadiness result;
  result.blocker_code = std::move(blocker);
  result.detail = std::move(detail);
  result.runtime = input.runtime;
  return result;
}

} // namespace

VulkanMattingReadiness
EvaluateVulkanMattingReadiness(const VulkanMattingReadinessInput &input) {
  if (!input.model_pack_selected) {
    return Blocked(input, kOpenVulkanMattingModelUnavailableReason,
                   "No schema-v2 ncnn_vulkan matting pack is selected.");
  }
  if (!input.model_contract_validated) {
    return Blocked(input, kOpenVulkanMattingModelInvalidReason,
                   "The selected ncnn_vulkan artifacts and exact tensor "
                   "metadata have not passed one-time integrity validation.");
  }
  if (!input.production_build_enabled) {
    return Blocked(input, kOpenVulkanMattingBuildDisabledReason,
                   "Production ncnn Vulkan matting is disabled in this build.");
  }
  if (!input.production_adapter_available) {
    return Blocked(
        input, kOpenVulkanMattingAdapterUnavailableReason,
        "No production ncnn adapter can import StudioCast's selected shared "
        "Vulkan device, queue, and buffers without a CPU bridge.");
  }
  if (!input.non_cpu_device_selected || !input.compute_queue_available ||
      !input.context_healthy) {
    return Blocked(input, kOpenVulkanMattingHardwareUnavailableReason,
                   "A healthy non-CPU Vulkan device and compute queue are "
                   "required.");
  }
  if (input.failure_latched ||
      input.failure != VulkanMattingRuntimeFailure::none) {
    return Blocked(input, FailureBlocker(input), input.failure_detail);
  }
  if (!input.session_initialized || !input.runtime.runtime_created ||
      !input.runtime.graph_loaded ||
      !input.runtime.persistent_resources_allocated) {
    return Blocked(input, kOpenVulkanMattingRuntimeFailureReason,
                   "The production runtime, graph, and persistent resources "
                   "are not prepared.");
  }
  if (!input.runtime.adapter_available || !input.runtime.production_adapter) {
    return Blocked(input, kOpenVulkanMattingAdapterUnavailableReason,
                   "The active runtime did not prove a production adapter.");
  }
  if (!input.runtime.device_identity_matches ||
      !input.runtime.shared_device_imported) {
    return Blocked(input, kOpenVulkanMattingDeviceMismatchReason,
                   "The runtime did not prove the exact shared Vulkan context "
                   "identity and generation.");
  }
  if (input.runtime.cpu_layers_used) {
    return Blocked(input, kOpenVulkanMattingCpuLayerReason,
                   "The ncnn graph used one or more CPU layers.");
  }
  if (input.runtime.cpu_readback_count != 0) {
    return Blocked(input, kOpenVulkanMattingCpuReadbackReason,
                   "The runtime reported a CPU input/alpha readback.");
  }
  if (!input.runtime.input_device_resident ||
      !input.runtime.alpha_device_resident ||
      !input.runtime.output_device_resident) {
    return Blocked(input, kOpenVulkanMattingResidencyReason,
                   "Input, alpha, and output residency is not proven.");
  }
  if (!input.runtime.queue_ownership_explicit ||
      !input.runtime.synchronous_completion ||
      input.runtime.completion_count == 0) {
    return Blocked(input, kOpenVulkanMattingSynchronizationReason,
                   "Queue ownership and synchronous completion are not "
                   "proven.");
  }
  if (!input.runtime.bounded_reusable_allocations ||
      input.runtime.persistent_allocation_count == 0 ||
      input.runtime.dynamic_allocation_count != 0) {
    return Blocked(input, kOpenVulkanMattingAllocationReason,
                   "Persistent bounded allocation reuse is not proven.");
  }
  if (!input.session_warmed || !input.runtime.warmup_complete ||
      input.runtime.warmup_inference_count != 1) {
    return Blocked(input, kOpenVulkanMattingWarmupReason,
                   "Exactly one successful device-resident warmup inference "
                   "is required.");
  }

  VulkanMattingReadiness result;
  result.production_ready = true;
  result.reason_code.clear();
  result.blocker_code.clear();
  result.detail = "Production device-resident Vulkan matting is ready.";
  result.runtime = input.runtime;
  return result;
}

std::string
FormatVulkanMattingReadiness(const VulkanMattingReadiness &readiness) {
  if (readiness.production_ready)
    return readiness.detail;
  std::string out = "[" + readiness.reason_code + "]";
  if (!readiness.blocker_code.empty())
    out += " [" + readiness.blocker_code + "]";
  if (!readiness.detail.empty())
    out += " " + readiness.detail;
  return out;
}

VulkanMattingRuntimeResult VulkanMattingRuntimeResult::Success() { return {}; }

VulkanMattingRuntimeResult
VulkanMattingRuntimeResult::Failure(VulkanMattingRuntimeFailure failure,
                                    std::string detail) {
  VulkanMattingRuntimeResult result;
  result.failure = failure;
  result.detail = std::move(detail);
  return result;
}

VulkanMattingRuntimeLifecycle::VulkanMattingRuntimeLifecycle(
    std::unique_ptr<VulkanMattingRuntime> runtime)
    : runtime_(std::move(runtime)) {}

VulkanMattingRuntimeLifecycle::~VulkanMattingRuntimeLifecycle() = default;

bool VulkanMattingRuntimeLifecycle::Fail(VulkanMattingRuntimeFailure failure,
                                         std::string detail, bool latch,
                                         std::string *error_out) {
  if (latch && latched_failure_ == VulkanMattingRuntimeFailure::none) {
    latched_failure_ = failure;
    latched_error_ = std::move(detail);
  }
  const std::string &reported =
      latched_failure_ == VulkanMattingRuntimeFailure::none ? detail
                                                            : latched_error_;
  if (error_out)
    *error_out = reported;
  return false;
}

bool VulkanMattingRuntimeLifecycle::AcceptResult(
    const VulkanMattingRuntimeResult &result, const char *operation,
    std::string *error_out) {
  if (result.ok())
    return true;
  return Fail(result.failure, PrefixError(operation, result.detail), true,
              error_out);
}

bool VulkanMattingRuntimeLifecycle::VerifyDeviceIdentity(
    const VulkanMattingRuntimeEvidence &evidence, std::string *error_out) {
  if (evidence.active_device.context_id != device_.context_id ||
      evidence.active_device.context_generation != device_.context_generation) {
    return Fail(VulkanMattingRuntimeFailure::context_generation_mismatch,
                "Open Vulkan matting runtime used a stale Vulkan context "
                "identity/generation.",
                true, error_out);
  }
  if (!evidence.device_identity_matches ||
      !SameDevice(device_, evidence.active_device)) {
    return Fail(VulkanMattingRuntimeFailure::device_identity_mismatch,
                "Open Vulkan matting runtime did not use StudioCast's exact "
                "physical device, logical device, queue, and allocation "
                "ownership domain.",
                true, error_out);
  }
  return true;
}

bool VulkanMattingRuntimeLifecycle::VerifySetupEvidence(
    const VulkanMattingRuntimeEvidence &evidence, std::string *error_out) {
  if (!evidence.adapter_available || !evidence.production_adapter ||
      !evidence.runtime_created || !evidence.graph_loaded ||
      !evidence.persistent_resources_allocated) {
    return Fail(VulkanMattingRuntimeFailure::invalid_graph,
                "Open Vulkan matting runtime setup completed without runtime, "
                "graph, and persistent-allocation evidence.",
                true, error_out);
  }
  if (!evidence.shared_device_imported || !evidence.queue_ownership_explicit) {
    return Fail(VulkanMattingRuntimeFailure::device_identity_mismatch,
                "Open Vulkan matting adapter did not import the exact shared "
                "device/queue ownership domain.",
                true, error_out);
  }
  if (!resources_.allow_cpu_layers && evidence.cpu_layers_used) {
    return Fail(VulkanMattingRuntimeFailure::cpu_layer_rejected,
                "Open Vulkan matting graph contains non-Vulkan/CPU layers.",
                true, error_out);
  }
  if (resources_.require_device_residency &&
      (!evidence.input_device_resident || !evidence.alpha_device_resident ||
       !evidence.output_device_resident)) {
    return Fail(VulkanMattingRuntimeFailure::residency_check_failed,
                "Open Vulkan matting runtime did not prove device-resident "
                "input, alpha, and output resources.",
                true, error_out);
  }
  if (evidence.cpu_readback_count != 0) {
    return Fail(VulkanMattingRuntimeFailure::cpu_readback_rejected,
                "Open Vulkan matting runtime reported a CPU input/alpha "
                "readback.",
                true, error_out);
  }
  if (!evidence.synchronous_completion) {
    return Fail(VulkanMattingRuntimeFailure::synchronization_failed,
                "Open Vulkan matting runtime did not prove explicit "
                "synchronous completion.",
                true, error_out);
  }
  if (!evidence.bounded_reusable_allocations ||
      evidence.persistent_allocation_count == 0 ||
      evidence.dynamic_allocation_count != 0) {
    return Fail(VulkanMattingRuntimeFailure::allocation_contract_failed,
                "Open Vulkan matting runtime did not prove bounded persistent "
                "allocation reuse with zero dynamic frame allocations.",
                true, error_out);
  }
  return VerifyDeviceIdentity(evidence, error_out);
}

bool VulkanMattingRuntimeLifecycle::VerifyExecutionEvidence(
    const VulkanMattingRuntimeEvidence &evidence, bool warmup,
    std::string *error_out) {
  if (!VerifySetupEvidence(evidence, error_out))
    return false;
  if (evidence.persistent_allocation_count != persistent_allocation_count_) {
    return Fail(VulkanMattingRuntimeFailure::allocation_contract_failed,
                "Open Vulkan matting persistent allocation count changed "
                "after setup.",
                true, error_out);
  }
  if (evidence.completion_count <= completion_count_) {
    return Fail(VulkanMattingRuntimeFailure::synchronization_failed,
                "Open Vulkan matting inference completed without advancing "
                "the synchronous completion counter.",
                true, error_out);
  }
  if (warmup) {
    if (!evidence.warmup_complete || evidence.warmup_inference_count != 1) {
      return Fail(VulkanMattingRuntimeFailure::execution_failed,
                  "Open Vulkan matting warmup did not execute exactly once.",
                  true, error_out);
    }
  } else if (evidence.inference_count == 0) {
    return Fail(VulkanMattingRuntimeFailure::execution_failed,
                "Open Vulkan matting execution evidence did not advance.", true,
                error_out);
  }
  completion_count_ = evidence.completion_count;
  return true;
}

bool VulkanMattingRuntimeLifecycle::Prepare(
    const VulkanMattingDeviceContext &device,
    const VulkanMattingGraphDescriptor &graph,
    const VulkanMattingPersistentResources &resources, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (failure_latched())
    return Fail(latched_failure_, latched_error_, false, error_out);
  if (prepared_)
    return true;
  if (prepare_attempted_) {
    return Fail(VulkanMattingRuntimeFailure::unavailable,
                "Open Vulkan matting runtime setup already failed; recreate "
                "the session before retrying.",
                false, error_out);
  }
  prepare_attempted_ = true;

  if (!runtime_) {
    return Fail(VulkanMattingRuntimeFailure::unavailable,
                "Open Vulkan matting runtime adapter is null.", false,
                error_out);
  }
  if (!CompleteGraphDescriptor(graph)) {
    return Fail(VulkanMattingRuntimeFailure::invalid_graph,
                "Open Vulkan matting graph descriptor requires param/bin "
                "paths, SHA-256 values, input/output blobs, converter "
                "name/version, and fp32/fp16 precision.",
                true, error_out);
  }
  std::size_t input_bytes = 0;
  std::size_t alpha_bytes = 0;
  if (!ExpectedF32Bytes(graph.input_n, graph.input_c, graph.input_h,
                        graph.input_w, &input_bytes) ||
      !ExpectedF32Bytes(graph.output_n, graph.output_c, graph.output_h,
                        graph.output_w, &alpha_bytes) ||
      input_bytes != resources.input_bytes ||
      alpha_bytes != resources.alpha_bytes) {
    return Fail(VulkanMattingRuntimeFailure::invalid_shape,
                "Open Vulkan matting persistent allocation sizes do not match "
                "the declared graph shapes.",
                true, error_out);
  }
  if (!device.ownership_domain || device.physical_device == 0 ||
      device.logical_device == 0 || device.queue == 0 ||
      device.context_id == 0 || device.context_generation == 0 ||
      !device.non_cpu_device_selected || !device.compute_queue_available ||
      !device.context_healthy) {
    return Fail(VulkanMattingRuntimeFailure::device_identity_mismatch,
                "Open Vulkan matting received an incomplete StudioCast Vulkan "
                "device context.",
                true, error_out);
  }

  device_ = device;
  graph_ = graph;
  resources_ = resources;
  if (!AcceptResult(runtime_->Initialize(device_), "runtime initialization",
                    error_out) ||
      !VerifyDeviceIdentity(runtime_->Evidence(), error_out) ||
      !AcceptResult(runtime_->LoadGraph(graph_), "graph load", error_out) ||
      !AcceptResult(runtime_->PreparePersistentResources(resources_),
                    "persistent allocation", error_out) ||
      !VerifySetupEvidence(runtime_->Evidence(), error_out)) {
    return false;
  }
  persistent_allocation_count_ =
      runtime_->Evidence().persistent_allocation_count;
  prepared_ = true;
  return true;
}

bool VulkanMattingRuntimeLifecycle::Warmup(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (failure_latched())
    return Fail(latched_failure_, latched_error_, false, error_out);
  if (warmed_)
    return true;
  if (!prepared_) {
    return Fail(VulkanMattingRuntimeFailure::unavailable,
                "Open Vulkan matting warmup requires a prepared runtime.",
                false, error_out);
  }
  if (warmup_attempted_) {
    return Fail(VulkanMattingRuntimeFailure::unavailable,
                "Open Vulkan matting warmup already failed; recreate the "
                "session before retrying.",
                false, error_out);
  }
  warmup_attempted_ = true;
  if (!AcceptResult(runtime_->Warmup(), "warmup", error_out))
    return false;
  const auto evidence = runtime_->Evidence();
  if (!VerifyExecutionEvidence(evidence, /*warmup=*/true, error_out))
    return false;
  warmed_ = true;
  return true;
}

bool VulkanMattingRuntimeLifecycle::VerifyBinding(
    const VulkanMattingBufferBinding &binding, const char *name, int expected_n,
    int expected_c, int expected_h, int expected_w, std::string *error_out) {
  std::size_t expected_bytes = 0;
  if (!ExpectedF32Bytes(expected_n, expected_c, expected_h, expected_w,
                        &expected_bytes) ||
      binding.n != expected_n || binding.c != expected_c ||
      binding.h != expected_h || binding.w != expected_w ||
      binding.byte_size < expected_bytes) {
    return Fail(VulkanMattingRuntimeFailure::invalid_shape,
                std::string("Open Vulkan matting ") + name +
                    " binding shape/size does not match the loaded graph.",
                true, error_out);
  }
  if (!binding.device_resident || !binding.device_local_memory ||
      binding.host_mapped || binding.buffer == 0) {
    return Fail(VulkanMattingRuntimeFailure::residency_check_failed,
                std::string("Open Vulkan matting ") + name +
                    " binding is not non-mapped device-local memory.",
                true, error_out);
  }
  if (binding.ownership_domain != device_.ownership_domain ||
      binding.logical_device != device_.logical_device) {
    return Fail(VulkanMattingRuntimeFailure::device_identity_mismatch,
                std::string("Open Vulkan matting ") + name +
                    " binding belongs to a different Vulkan device or "
                    "allocation domain.",
                true, error_out);
  }
  if (binding.context_id != device_.context_id ||
      binding.context_generation != device_.context_generation) {
    return Fail(VulkanMattingRuntimeFailure::context_generation_mismatch,
                std::string("Open Vulkan matting ") + name +
                    " binding belongs to a stale Vulkan context "
                    "identity/generation.",
                true, error_out);
  }
  if (!binding.access_synchronized || !binding.runtime_queue_ownership) {
    return Fail(VulkanMattingRuntimeFailure::synchronization_failed,
                std::string("Open Vulkan matting ") + name +
                    " binding lacks explicit synchronization/queue ownership.",
                true, error_out);
  }
  return true;
}

bool VulkanMattingRuntimeLifecycle::Run(
    const VulkanMattingBufferBinding &input,
    const VulkanMattingBufferBinding &alpha_output, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (failure_latched())
    return Fail(latched_failure_, latched_error_, false, error_out);
  if (!warmed_) {
    return Fail(VulkanMattingRuntimeFailure::unavailable,
                "Open Vulkan matting run requires successful one-time warmup.",
                false, error_out);
  }
  if (!VerifyBinding(input, "input", graph_.input_n, graph_.input_c,
                     graph_.input_h, graph_.input_w, error_out) ||
      !VerifyBinding(alpha_output, "alpha output", graph_.output_n,
                     graph_.output_c, graph_.output_h, graph_.output_w,
                     error_out)) {
    return false;
  }
  if (!AcceptResult(runtime_->Run(input, alpha_output), "execution", error_out))
    return false;
  const auto evidence = runtime_->Evidence();
  return VerifyExecutionEvidence(evidence, /*warmup=*/false, error_out);
}

bool VulkanMattingRuntimeLifecycle::LatchExternalFailure(
    VulkanMattingRuntimeFailure failure, std::string detail,
    std::string *error_out) {
  if (failure == VulkanMattingRuntimeFailure::none) {
    if (error_out)
      *error_out = "Open Vulkan matting cannot latch a 'none' failure.";
    return false;
  }
  return Fail(failure, std::move(detail), true, error_out);
}

bool VulkanMattingRuntimeLifecycle::available() const {
  return prepared_ && warmed_ && !failure_latched();
}

bool VulkanMattingRuntimeLifecycle::prepared() const { return prepared_; }

bool VulkanMattingRuntimeLifecycle::warmed() const { return warmed_; }

bool VulkanMattingRuntimeLifecycle::failure_latched() const {
  return latched_failure_ != VulkanMattingRuntimeFailure::none;
}

VulkanMattingRuntimeFailure
VulkanMattingRuntimeLifecycle::latched_failure() const {
  return latched_failure_;
}

const std::string &VulkanMattingRuntimeLifecycle::latched_error() const {
  return latched_error_;
}

VulkanMattingRuntimeEvidence VulkanMattingRuntimeLifecycle::Evidence() const {
  return runtime_ ? runtime_->Evidence() : VulkanMattingRuntimeEvidence{};
}

std::unique_ptr<VulkanMattingRuntime>
CreateUnavailableVulkanMattingRuntime(std::string reason) {
  return std::make_unique<UnavailableVulkanMattingRuntime>(std::move(reason));
}

bool ProductionVulkanMattingBuildEnabled() {
#if defined(STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING) &&                          \
    STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  return true;
#else
  return false;
#endif
}

bool ProductionVulkanMattingAdapterAvailable() {
  // ncnn's public Vulkan runtime owns/chooses its Vulkan device. The CPU-Mat
  // spike cannot import StudioCast's live context or buffers, so it is never a
  // production adapter.
  return false;
}

std::unique_ptr<VulkanMattingRuntime> CreateProductionVulkanMattingRuntime() {
  const char *reason = ProductionVulkanMattingBuildEnabled()
                           ? "Production ncnn is linked, but no reviewed "
                             "shared-device Vulkan adapter is available."
                           : "Production ncnn Vulkan matting is disabled in "
                             "this build.";
  return CreateUnavailableVulkanMattingRuntime(reason);
}

} // namespace studiocast::open_vulkan
