#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "core/open_video/fastdvdnet_denoiser.h"
#include "core/open_video/gaze_correction_eye_contact.h"
#include "core/open_video/yunet_face_detector.h"

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
  ok &= Require(ValidateYunetInputTensorContract({1, 3, 320, 320}, false,
                                                 320, 320, &error),
                "matching fixed YuNet geometry should validate: " + error);
  ok &= Require(ValidateYunetInputTensorContract({1, 3, -1, -1}, false,
                                                 320, 320, &error),
                "dynamic YuNet geometry should accept the manifest runtime "
                "shape: " + error);
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

} // namespace studiocast::tests
