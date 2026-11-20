#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "core/open_video/fastdvdnet_denoiser.h"
#include "core/open_video/gaze_correction_eye_contact.h"
#include "core/open_video/yunet_face_detector.h"
#include "core/video/open_vulkan_eye_contact.h"
#include "core/video/open_vulkan_video_noise_removal.h"

namespace {

bool Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool RequireShape(const std::vector<int64_t> &got,
                  const std::vector<int64_t> &want,
                  const std::string &message) {
  if (got == want)
    return true;
  std::cerr << message << "\n";
  return false;
}

} // namespace

namespace studiocast::tests {

bool TestFastDvdnetDenoiseTensorContractIsDeclared() {
  studiocast::open_video::FastDvdnetDenoiser denoiser;
  const auto &contract = denoiser.tensor_io_contract();

  bool ok = true;
  ok &= Require(contract.adapter_id == "fastdvdnet",
                "unexpected denoise adapter id");
  ok &= Require(contract.model_family == "FastDVDnet",
                "unexpected denoise model family");
  ok &= Require(contract.inputs.size() == 2,
                "FastDVDnet contract should declare two inputs");

  if (contract.inputs.size() == 2) {
    ok &= Require(contract.inputs[0].role == "temporal_rgb_window",
                  "unexpected temporal input role");
    ok &= Require(contract.inputs[0].layout == "NCHW",
                  "unexpected temporal input layout");
    ok &= RequireShape(contract.inputs[0].shape, {1, 15, -1, -1},
                       "unexpected temporal input shape");

    ok &= Require(contract.inputs[1].role == "noise_map",
                  "unexpected noise input role");
    ok &= RequireShape(contract.inputs[1].shape, {1, 1, -1, -1},
                       "unexpected noise input shape");
  }

  ok &=
      Require(contract.output.role == "denoised_rgb", "unexpected output role");
  ok &= RequireShape(contract.output.shape, {1, 3, -1, -1},
                     "unexpected output shape");

  ok &= Require(contract.temporal.window_frames == 5,
                "unexpected temporal window");
  ok &= Require(contract.temporal.history_frames == 3,
                "unexpected temporal history");
  ok &= Require(contract.temporal.repeated_future_frames == 2,
                "unexpected repeated future frame count");
  ok &= Require(contract.temporal.causal, "contract should be causal");

  ok &= Require(contract.supports_cpu_tensor_io,
                "contract should support CPU tensor IO");
  ok &= Require(contract.supports_cuda_device_tensor_io,
                "contract should support CUDA device tensor IO");
  ok &= Require(contract.requires_cpu_preprocess,
                "contract should declare CPU preprocess");
  ok &= Require(contract.requires_cpu_postprocess,
                "contract should declare CPU postprocess");
  ok &= Require(contract.requires_output_device_to_cpu_for_postprocess,
                "contract should declare denoised tensor readback");
  return ok;
}

bool TestYunetFaceDetectionCpuTensorTailContractIsDeclared() {
  studiocast::open_video::YunetFaceDetector detector;
  const auto status = detector.runtime_status();

  bool ok = true;
  ok &= Require(status.uses_cpu_preprocess,
                "YuNet should declare CPU preprocess");
  ok &=
      Require(status.uses_cpu_tensor_io, "YuNet should declare CPU tensor I/O");
  ok &= Require(status.uses_cpu_postprocess,
                "YuNet should declare CPU postprocess");
  ok &= Require(!status.device_resident_gpu_path,
                "YuNet must not claim a device-resident GPU path");
  ok &= Require(status.summary.find("not a device-resident GPU path") !=
                    std::string::npos,
                "YuNet summary should reject hidden GPU-resident claims");
  return ok;
}

bool TestYunetExplicitVulkanProviderAndTensorPolicyIsFailClosed() {
  using studiocast::open_video::ValidateYunetInputTensorContract;
  using studiocast::open_video::YunetOrtSessionOptions;
  using studiocast::open_video::YunetProviderPolicy;

  const auto vulkan_policy =
      YunetOrtSessionOptions(YunetProviderPolicy::cpu_only);
  const auto cuda_policy =
      YunetOrtSessionOptions(YunetProviderPolicy::prefer_cuda);
  std::string error;
  bool ok = Require(!vulkan_policy.prefer_cuda,
                    "explicit Vulkan YuNet policy must be CPU-only") &&
            Require(!vulkan_policy.enable_tensorrt,
                    "explicit Vulkan YuNet policy must not append TensorRT") &&
            Require(cuda_policy.prefer_cuda,
                    "existing Open CUDA YuNet policy should remain compatible");
  ok &= Require(ValidateYunetInputTensorContract({1, 3, 320, 320}, false, 320,
                                                 320, &error),
                "matching fixed YuNet geometry should validate: " + error);
  ok &= Require(
      ValidateYunetInputTensorContract({1, 3, -1, -1}, false, 320, 320, &error),
      "dynamic YuNet geometry should accept the manifest runtime "
      "shape: " +
          error);
  ok &= Require(
      !ValidateYunetInputTensorContract({1, 3, 640, 640}, false, 320, 320,
                                        &error) &&
          error.find("[vulkan_auto_frame_yunet_tensor_geometry_mismatch]") !=
              std::string::npos &&
          error.find("640x640") != std::string::npos &&
          error.find("320x320") != std::string::npos,
      "fixed installed-graph/manifest mismatch must fail with exact evidence");
  return ok;
}

bool TestOpenVideoEyeContactCpuTensorTailContractIsDeclared() {
  studiocast::open_video::GazeCorrectionEyeContact eye_contact;
  const auto status = eye_contact.runtime_status();

  bool ok = true;
  ok &= Require(status.uses_cpu_face_detection,
                "eye contact should declare CPU face detection");
  ok &= Require(status.uses_cpu_landmarks,
                "eye contact should declare CPU landmarks");
  ok &= Require(status.uses_cpu_preprocess,
                "eye contact should declare CPU preprocess");
  ok &= Require(status.uses_cpu_tensor_io,
                "eye contact should declare CPU tensor I/O");
  ok &= Require(status.uses_cpu_postprocess,
                "eye contact should declare CPU postprocess");
  ok &= Require(!status.device_resident_gpu_path,
                "eye contact must not claim a device-resident GPU path");
  ok &= Require(status.summary.find("not a device-resident GPU path") !=
                    std::string::npos,
                "eye contact summary should reject hidden GPU-resident claims");
  return ok;
}

bool TestOpenVulkanEyeContactCapabilityFactsAreFailClosed() {
  using studiocast::video::CurrentOpenVulkanEyeContactFacts;
  using studiocast::video::EvaluateOpenVulkanEyeContactReadiness;
  using studiocast::video::FormatOpenVulkanEyeContactReadiness;
  using studiocast::video::kOpenVulkanEyeContactRuntimeUnavailableReason;
  using studiocast::video::kOpenVulkanEyeContactUnavailableReason;

  const auto facts = CurrentOpenVulkanEyeContactFacts();
  const auto readiness = EvaluateOpenVulkanEyeContactReadiness(facts);
  const std::string formatted = FormatOpenVulkanEyeContactReadiness(readiness);

  bool ok = true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  ok &= Require(facts.backend_compiled,
                "Vulkan-on build should report the backend compiled fact");
#else
  ok &= Require(!facts.backend_compiled,
                "Vulkan-off build should report the backend disabled fact");
#endif
#if STUDIOCAST_ENABLE_OPEN_VULKAN && STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  ok &= Require(facts.vulkan_inference_provider_available,
                "the production ncnn build should expose its provider");
#if STUDIOCAST_HAVE_DLIB
  ok &= Require(facts.live_stage_implemented &&
                    facts.production_adapter_available,
                "the ncnn/dlib build should expose the resident eye stage");
#else
  ok &= Require(!facts.live_stage_implemented &&
                    !facts.production_adapter_available,
                "the eye stage requires its compiled landmark dependency");
#endif
#else
  ok &= Require(
      !facts.live_stage_implemented && !facts.production_adapter_available &&
          !facts.vulkan_inference_provider_available,
      "missing production ncnn must not expose an eye adapter/provider");
#endif
  ok &=
      Require(!facts.shared_device_imported && !facts.queue_ownership_explicit,
              "eye contact must not claim shared device/queue ownership");
  ok &=
      Require(!facts.model_pack_selected && !facts.artifact_contract_validated,
              "ONNX template presence must not satisfy Vulkan artifacts");
  ok &= Require(!facts.device_resident_analysis &&
                    !facts.device_resident_tensor_io,
                "CPU analysis/tensors must not claim device residency");
  ok &= Require(!facts.selectable_cpu_fallback,
                "internal Open Video CPU work is not a selectable fallback");
  ok &= Require(facts.dispatch_count == 0 && facts.cpu_readback_count == 0 &&
                    facts.cpu_fallback_count == 0,
                "compile-time eye facts must perform no frame work");
  ok &= Require(!readiness.production_ready &&
                    readiness.reason_code ==
                        kOpenVulkanEyeContactUnavailableReason &&
                    readiness.blocker_code ==
                        kOpenVulkanEyeContactRuntimeUnavailableReason,
                "eye contact readiness must keep exact nested blockers");
  ok &= Require(
      formatted.find("[open_vulkan_eye_contact_unavailable]") !=
              std::string::npos &&
          formatted.find("[open_vulkan_eye_contact_runtime_unavailable]") !=
              std::string::npos &&
          formatted.find("converted artifacts") != std::string::npos &&
          formatted.find("parity evidence") != std::string::npos,
      "formatted readiness must identify missing prepared model/context "
      "evidence");

  using Facts = studiocast::video::OpenVulkanEyeContactCapabilityFacts;
  Facts complete;
  complete.backend_compiled = complete.live_stage_implemented = true;
  complete.production_adapter_available =
      complete.vulkan_inference_provider_available = true;
  complete.non_cpu_device_selected = complete.compute_queue_available =
      complete.context_healthy = true;
  complete.shared_device_imported = complete.queue_ownership_explicit = true;
  complete.model_pack_selected = complete.artifact_contract_validated = true;
  complete.device_resident_analysis = complete.device_resident_tensor_io = true;
  complete.warmup_complete = complete.bounded_reusable_allocations = true;
  complete.synchronization_contract_validated = complete.parity_validated =
      true;
  complete.dispatch_count = 1;
  const auto ready = EvaluateOpenVulkanEyeContactReadiness(complete);
  ok &= Require(
      ready.production_ready &&
          ready.reason_code ==
              studiocast::video::kOpenVulkanEyeContactProductionReadyReason &&
          ready.blocker_code.empty() && ready.detail.empty(),
      "complete synthetic eye evidence must return the exact success "
      "attestation");
  unsigned index = 0;
  for (auto member :
       {&Facts::backend_compiled, &Facts::live_stage_implemented,
        &Facts::production_adapter_available,
        &Facts::vulkan_inference_provider_available,
        &Facts::non_cpu_device_selected, &Facts::compute_queue_available,
        &Facts::context_healthy, &Facts::shared_device_imported,
        &Facts::queue_ownership_explicit, &Facts::model_pack_selected,
        &Facts::artifact_contract_validated, &Facts::device_resident_analysis,
        &Facts::device_resident_tensor_io, &Facts::warmup_complete,
        &Facts::bounded_reusable_allocations,
        &Facts::synchronization_contract_validated, &Facts::parity_validated}) {
    Facts missing = complete;
    missing.*member = false;
    ok &= Require(
        !EvaluateOpenVulkanEyeContactReadiness(missing).production_ready,
        "missing eye readiness prerequisite " + std::to_string(index++));
  }
  for (auto member : {&Facts::dispatch_count, &Facts::cpu_readback_count,
                      &Facts::cpu_fallback_count}) {
    Facts invalid = complete;
    invalid.*member = member == &Facts::dispatch_count ? 0 : 1;
    ok &= Require(
        !EvaluateOpenVulkanEyeContactReadiness(invalid).production_ready,
        "invalid eye execution/transfer evidence must fail closed");
  }
  return ok;
}

bool TestOpenVulkanVideoNoiseRemovalCapabilityFactsAreFailClosed() {
  using studiocast::video::CurrentOpenVulkanVideoNoiseRemovalFacts;
  using studiocast::video::EvaluateOpenVulkanVideoNoiseRemovalReadiness;
  using studiocast::video::FormatOpenVulkanVideoNoiseRemovalReadiness;
  using studiocast::video::kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason;
  using studiocast::video::kOpenVulkanVideoNoiseRemovalUnavailableReason;

  const auto facts = CurrentOpenVulkanVideoNoiseRemovalFacts();
  const auto readiness = EvaluateOpenVulkanVideoNoiseRemovalReadiness(facts);
  const std::string formatted =
      FormatOpenVulkanVideoNoiseRemovalReadiness(readiness);

  bool ok = true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  ok &= Require(facts.backend_compiled,
                "Vulkan-on build should report the backend compiled fact");
#else
  ok &= Require(!facts.backend_compiled,
                "Vulkan-off build should report the backend disabled fact");
#endif
#if STUDIOCAST_ENABLE_OPEN_VULKAN && STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  ok &= Require(
      facts.live_stage_implemented && facts.production_adapter_available &&
          facts.vulkan_inference_provider_available,
      "the production ncnn build should expose the resident denoiser");
#else
  ok &= Require(
      !facts.live_stage_implemented && !facts.production_adapter_available &&
          !facts.vulkan_inference_provider_available,
      "missing production ncnn must not expose a denoise adapter/provider");
#endif
  ok &=
      Require(!facts.shared_device_imported && !facts.queue_ownership_explicit,
              "video denoise must not claim shared device/queue ownership");
  ok &=
      Require(!facts.model_pack_selected && !facts.artifact_contract_validated,
              "an ONNX-only FastDVDnet pack must not satisfy Vulkan "
              "artifacts");
  ok &= Require(!facts.fully_device_resident_tensor_io &&
                    !facts.device_resident_preprocess &&
                    !facts.device_resident_postprocess,
                "host tensor packing/pre/post must not claim residency");
  ok &= Require(!facts.warmup_complete &&
                    !facts.synchronization_contract_validated &&
                    !facts.bounded_reusable_allocations,
                "missing runtime must not claim warmup/sync/allocation proof");
  ok &= Require(!facts.temporal_history_device_resident &&
                    !facts.temporal_history_bounded &&
                    !facts.history_reset_on_disable &&
                    !facts.history_reset_on_reconfigure &&
                    !facts.capture_sequence_discontinuity_reset,
                "host FastDVDnet history semantics do not prove a Vulkan "
                "temporal contract");
  ok &= Require(
      !facts.parity_validated && !facts.selectable_cpu_fallback,
      "compile-time availability must not infer parity or CPU fallback");
  ok &= Require(
      facts.dispatch_count == 0 && facts.temporal_history_reset_count == 0 &&
          facts.cpu_readback_count == 0 && facts.cpu_fallback_count == 0,
      "compile-time video denoise facts must perform no frame work");
  ok &= Require(!readiness.production_ready &&
                    readiness.reason_code ==
                        kOpenVulkanVideoNoiseRemovalUnavailableReason &&
                    readiness.blocker_code ==
                        kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason,
                "video denoise readiness must keep exact nested blockers");
  ok &= Require(
      formatted.find("[open_vulkan_video_noise_removal_unavailable]") !=
              std::string::npos &&
          formatted.find(
              "[open_vulkan_video_noise_removal_runtime_unavailable]") !=
              std::string::npos &&
          formatted.find("resident five-frame") != std::string::npos &&
          formatted.find("converted FastDVDnet") != std::string::npos &&
          formatted.find("parity evidence") != std::string::npos,
      "formatted readiness must identify missing prepared model/history "
      "evidence");
  return ok;
}

bool TestOpenVulkanVideoNoiseRemovalTemporalReadinessGatesAreExact() {
  using Facts = studiocast::video::OpenVulkanVideoNoiseRemovalCapabilityFacts;
  using studiocast::video::EvaluateOpenVulkanVideoNoiseRemovalReadiness;

  Facts facts;
  facts.backend_compiled = true;
  facts.live_stage_implemented = true;
  facts.production_adapter_available = true;
  facts.vulkan_inference_provider_available = true;
  facts.non_cpu_device_selected = true;
  facts.compute_queue_available = true;
  facts.context_healthy = true;
  facts.shared_device_imported = true;
  facts.queue_ownership_explicit = true;
  facts.model_pack_selected = true;
  facts.artifact_contract_validated = true;
  facts.fully_device_resident_tensor_io = true;
  facts.device_resident_preprocess = true;
  facts.device_resident_postprocess = true;
  facts.warmup_complete = true;
  facts.synchronization_contract_validated = true;
  facts.bounded_reusable_allocations = true;
  facts.temporal_history_device_resident = true;
  facts.temporal_history_bounded = true;
  facts.history_reset_on_disable = true;
  facts.history_reset_on_reconfigure = true;
  facts.capture_sequence_discontinuity_reset = true;
  facts.parity_validated = true;
  facts.dispatch_count = 1;

  const auto ready = EvaluateOpenVulkanVideoNoiseRemovalReadiness(facts);
  bool ok =
      Require(ready.production_ready &&
                  ready.reason_code ==
                      studiocast::video::
                          kOpenVulkanVideoNoiseRemovalProductionReadyReason &&
                  ready.blocker_code.empty() && ready.detail.empty(),
              "complete synthetic evidence should return the exact success "
              "attestation");
  const auto require_gate = [&](bool Facts::*member,
                                const std::string &message) {
    Facts missing = facts;
    missing.*member = false;
    return Require(
        !EvaluateOpenVulkanVideoNoiseRemovalReadiness(missing).production_ready,
        message);
  };
  ok &= require_gate(&Facts::bounded_reusable_allocations,
                     "bounded reusable allocations must gate readiness");
  ok &= require_gate(&Facts::temporal_history_device_resident,
                     "resident temporal history must gate readiness");
  ok &= require_gate(&Facts::temporal_history_bounded,
                     "bounded temporal history must gate readiness");
  ok &= require_gate(&Facts::history_reset_on_disable,
                     "disable/zero-strength reset must gate readiness");
  ok &= require_gate(&Facts::history_reset_on_reconfigure,
                     "geometry/model reconfigure reset must gate readiness");
  ok &= require_gate(&Facts::capture_sequence_discontinuity_reset,
                     "capture-sequence discontinuity reset must gate "
                     "readiness");
  ok &= require_gate(&Facts::parity_validated,
                     "temporal/output parity must gate readiness");
  unsigned index = 0;
  for (auto member :
       {&Facts::backend_compiled, &Facts::live_stage_implemented,
        &Facts::production_adapter_available,
        &Facts::vulkan_inference_provider_available,
        &Facts::non_cpu_device_selected, &Facts::compute_queue_available,
        &Facts::context_healthy, &Facts::shared_device_imported,
        &Facts::queue_ownership_explicit, &Facts::model_pack_selected,
        &Facts::artifact_contract_validated,
        &Facts::fully_device_resident_tensor_io,
        &Facts::device_resident_preprocess, &Facts::device_resident_postprocess,
        &Facts::warmup_complete, &Facts::synchronization_contract_validated})
    ok &= require_gate(member, "missing denoise readiness prerequisite " +
                                   std::to_string(index++));
  Facts unexecuted = facts;
  unexecuted.dispatch_count = 0;
  ok &= Require(!EvaluateOpenVulkanVideoNoiseRemovalReadiness(unexecuted)
                     .production_ready,
                "unexecuted denoise must not claim ready");

  Facts readback = facts;
  readback.cpu_readback_count = 1;
  ok &= Require(
      !EvaluateOpenVulkanVideoNoiseRemovalReadiness(readback).production_ready,
      "a CPU readback must prevent Vulkan-native readiness");
  Facts fallback = facts;
  fallback.cpu_fallback_count = 1;
  ok &= Require(
      !EvaluateOpenVulkanVideoNoiseRemovalReadiness(fallback).production_ready,
      "a CPU fallback must prevent Vulkan-native readiness");
  return ok;
}

} // namespace studiocast::tests
