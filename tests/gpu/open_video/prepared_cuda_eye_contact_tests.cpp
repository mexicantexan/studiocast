#include "core/open_video/native_prepared_cuda_eye_contact_runtime.h"
#include "core/open_video/prepared_cuda_eye_contact.h"

#include <cstdlib>
#include <iostream>

using namespace studiocast;
using namespace studiocast::open_video;
namespace {
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::cerr << "CHECK failed: " #x << '\n';                                \
      return false;                                                            \
    }                                                                          \
  } while (0)
ValidatedModelArtifactIdentity Identity(unsigned char value) {
  ValidatedModelArtifactIdentity id;
  id.registry_generation = value;
  id.model_id_sha256.fill(value);
  id.manifest_sha256.fill(value);
  id.artifact_sha256.fill(value);
  id.artifact_set_sha256.fill(value);
  return id;
}
struct Analysis final : IPreparedCudaEyeContactAnalysis {
  ValidatedModelArtifactIdentity y = Identity(1), d = Identity(2);
  PreparedCudaEyeContactAnalysisResult next{};
  PreparedCudaEyeContactStatus landmark =
      PreparedCudaEyeContactStatus::executed;
  cuda::CudaTensor tensor;
  int prepare = 0, detect = 0, landmarks = 0, shutdown = 0;
  Analysis() {
    tensor.ptr = 1;
    tensor.pitch = tensor.bytes = 136 * sizeof(float);
    tensor.n = 1;
    tensor.c = 1;
    tensor.h = 1;
    tensor.w = 136;
  }
  PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactSetup &) noexcept override {
    ++prepare;
    return PreparedCudaEyeContactStatus::executed;
  }
  const ValidatedModelArtifactIdentity &artifact() const noexcept override {
    return y;
  }
  const ValidatedModelArtifactIdentity &
  landmark_artifact() const noexcept override {
    return d;
  }
  PreparedCudaEyeContactAnalysisResult
  Detect(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
         std::uint64_t, FrameAnalysisCache *,
         const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept
      override {
    ++detect;
    return next;
  }
  PreparedCudaEyeContactStatus
  Landmarks(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
            const PreparedCudaFaceBox &, std::uint64_t,
            const cuda::CudaTensor **out) noexcept override {
    ++landmarks;
    *out =
        landmark == PreparedCudaEyeContactStatus::executed ? &tensor : nullptr;
    return landmark;
  }
  PreparedCudaEyeContactStatus Shutdown() noexcept override {
    ++shutdown;
    return PreparedCudaEyeContactStatus::shutdown;
  }
};
struct Runtime final : IPreparedCudaEyeContactRuntime {
  PreparedCudaEyeContactStatus prepare_status =
                                   PreparedCudaEyeContactStatus::executed,
                               shutdown_status =
                                   PreparedCudaEyeContactStatus::shutdown;
  PreparedCudaEyeContactRuntimeFrameResult next{};
  int prepare = 0, execute = 0, shutdown = 0;
  PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactRuntimeConfig &) noexcept override {
    ++prepare;
    return prepare_status;
  }
  PreparedCudaEyeContactRuntimeFrameResult
  Execute(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
          const cuda::CudaTensor &, float) noexcept override {
    ++execute;
    return next;
  }
  PreparedCudaEyeContactStatus Shutdown() noexcept override {
    ++shutdown;
    return shutdown_status;
  }
};
PreparedCudaEyeContactBinding Binding() {
  PreparedCudaEyeContactBinding b;
  b.driver = reinterpret_cast<cuda::CudaDriverApi *>(1);
  b.owner_context = reinterpret_cast<cuda::CUcontext>(2);
  b.stream = reinterpret_cast<cuda::CUstream>(3);
  b.context.physical_device.uuid[0] = 1;
  b.context.physical_device.pci_bus_id[0] = '1';
  b.context.physical_device.pci_bus_id_length = 1;
  b.context.retained_context_id = 4;
  b.context.logical_generation = 5;
  b.stream_identity = 6;
  b.owner_identity = 7;
  b.configuration_generation = 8;
  b.model_generation = 9;
  return b;
}
PreparedCudaEyeContactSetup Setup(FrameAnalysisCache *cache) {
  PreparedCudaEyeContactSetup s;
  s.binding = Binding();
  s.models.yunet = Identity(1);
  s.models.landmarks = Identity(2);
  s.models.left = Identity(3);
  s.models.right = Identity(4);
  s.models.left_path = "left.onnx";
  s.models.right_path = "right.onnx";
  s.frame_width = 640;
  s.frame_height = 480;
  s.strength = 50;
  s.resource_budget_bytes = 1 << 20;
  s.analysis_cache = cache;
  return s;
}
bool ContractsAndBehavior() {
  onnx::OrtSessionInfo info;
  info.input_names = {"eye_rgb:0", "anchors:0", "angles:0"};
  info.output_names = {"rgb:0"};
  info.input_shapes = {{1, 3, 48, 64}, {1, 12, 48, 64}, {1, 2}};
  info.output_shapes = {{1, 3, 48, 64}};
  info.input_elem_types = {1, 1, 1};
  info.output_elem_types = {1};
  CHECK(ValidatePreparedEyeContactSessionMetadata(info));
  info.input_elem_types[0] = 10;
  CHECK(!ValidatePreparedEyeContactSessionMetadata(info));
  FrameAnalysisCache cache;
  Analysis analysis;
  Runtime runtime;
  PreparedCudaEyeContact eye(&analysis, &runtime);
  PreparedCudaEyeContactHandle h;
  auto setup = Setup(&cache);
  std::string error;
  CHECK(eye.Prepare(setup, &h, &error));
  CHECK(h.id && analysis.prepare == 1 && runtime.prepare == 1);
  cuda::CudaImage input;
  input.ptr = 11;
  input.pitch = 1920;
  input.w = 640;
  input.h = 480;
  input.format = cuda::PixelFormatGpu::rgb_u8;
  analysis.next.status = PreparedCudaEyeContactStatus::executed;
  analysis.next.found = false;
  analysis.next.enqueued_readbacks = 12;
  analysis.next.completed_readbacks = 12;
  analysis.next.completion_waits = 1;
  analysis.next.cpu_tail_boundaries = 1;
  auto r = eye.Execute(h, setup.binding, 1, input);
  CHECK(r.status == PreparedCudaEyeContactStatus::preserved_input &&
        r.output == &input && r.valid_input_preserved && runtime.execute == 0 &&
        r.enqueued_readbacks == 12 && r.completed_readbacks == 12 &&
        r.completion_waits == 1 && r.cpu_tail_boundaries == 1);
  analysis.next.found = true;
  runtime.next.status = PreparedCudaEyeContactStatus::executed;
  runtime.next.output = &input;
  runtime.next.inference_attempts = 2;
  runtime.next.inference_successes = 2;
  r = eye.Execute(h, setup.binding, 2, input);
  CHECK(r.status == PreparedCudaEyeContactStatus::executed &&
        analysis.landmarks == 1 && runtime.execute == 1 &&
        r.inference_attempts == 2);
  runtime.next.status = PreparedCudaEyeContactStatus::inference_failed;
  runtime.next.output = &input;
  runtime.next.pending_work = false;
  r = eye.Execute(h, setup.binding, 3, input);
  CHECK(r.status == PreparedCudaEyeContactStatus::inference_failed &&
        r.valid_input_preserved && r.output == &input);
  auto stale = setup.binding;
  ++stale.model_generation;
  r = eye.Execute(h, stale, 4, input);
  CHECK(r.status == PreparedCudaEyeContactStatus::stale_binding &&
        runtime.execute == 2);
  CHECK(eye.Shutdown(&error));
  CHECK(analysis.shutdown == 1 && runtime.shutdown == 1);
  return true;
}
bool FailureAndDisabled() {
  FrameAnalysisCache cache;
  Analysis analysis;
  Runtime runtime;
  PreparedCudaEyeContact eye(&analysis, &runtime);
  auto setup = Setup(&cache);
  setup.strength = 0;
  PreparedCudaEyeContactHandle h;
  std::string error;
  CHECK(eye.Prepare(setup, &h, &error));
  cuda::CudaImage input;
  input.w = 640;
  input.h = 480;
  input.format = cuda::PixelFormatGpu::rgb_u8;
  auto r = eye.Execute(h, setup.binding, 1, input);
  CHECK(r.valid_input_preserved && analysis.detect == 0 &&
        runtime.execute == 0);
  CHECK(eye.Shutdown(&error));
  runtime.prepare_status = PreparedCudaEyeContactStatus::provider_mismatch;
  CHECK(!eye.Prepare(Setup(&cache), &h, &error));
  CHECK(!eye.ready());
  return true;
}
} // namespace
int main() {
  return ContractsAndBehavior() && FailureAndDisabled() ? EXIT_SUCCESS
                                                        : EXIT_FAILURE;
}
