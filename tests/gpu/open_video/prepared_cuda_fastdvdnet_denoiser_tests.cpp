#include "core/open_video/prepared_cuda_fastdvdnet_denoiser.h"

#include <cmath>
#include <iostream>

namespace {
using namespace studiocast;
using namespace studiocast::open_video;

bool Check(bool value, const char *message) {
  if (!value) std::cerr << "FAIL: " << message << '\n';
  return value;
}

struct FakeRuntime final : IPreparedCudaFastDvdnetRuntime {
  PreparedCudaFastDvdnetRuntimePrepareResult prepare{
      PreparedCudaFastDvdnetStatus::ready, 44, {4096, 1024, 0, 0, 8, 1},
      true, true, true, true, true};
  PreparedCudaFastDvdnetRuntimeFrameResult frame{
      PreparedCudaFastDvdnetStatus::executed,
      PreparedCudaFastDvdnetOperation::none, {}, {}, 11, 17, true};
  PreparedCudaFastDvdnetTemporalPlan last{};
  float sigma = 0;
  int prepares = 0, executes = 0, resets = 0, shutdowns = 0;
  bool first_shutdown_pending = false;

  PreparedCudaFastDvdnetRuntimePrepareResult
  Prepare(const PreparedCudaFastDvdnetConfig &) noexcept override {
    ++prepares; return prepare;
  }
  PreparedCudaFastDvdnetRuntimeFrameResult Execute(
      std::uint64_t, const PreparedCudaFastDvdnetTemporalPlan &p,
      const cuda::CudaImage &, cuda::CudaImage *, float s) noexcept override {
    ++executes; last = p; sigma = s; return frame;
  }
  PreparedCudaFastDvdnetStatus ResetTemporalState(std::uint64_t) noexcept override {
    ++resets; return PreparedCudaFastDvdnetStatus::ready;
  }
  PreparedCudaFastDvdnetShutdownSnapshot Shutdown(std::uint64_t) noexcept override {
    ++shutdowns;
    if (first_shutdown_pending && shutdowns == 1)
      return {PreparedCudaFastDvdnetStatus::completion_pending, 1, true, false};
    return {PreparedCudaFastDvdnetStatus::shutdown,
            static_cast<std::uint64_t>(shutdowns), true, true};
  }
  PreparedCudaFastDvdnetRuntimeTelemetry Telemetry() const noexcept override {
    return {static_cast<std::uint64_t>(prepares), 0, 0, 1, 1,
            static_cast<std::uint64_t>(executes * 2)};
  }
};

ValidatedModelArtifactIdentity Artifact(std::uint8_t seed = 1) {
  ValidatedModelArtifactIdentity a; a.registry_generation = 2;
  a.model_id_sha256.fill(seed); a.manifest_sha256.fill(seed + 1);
  a.artifact_sha256.fill(seed + 2); a.artifact_set_sha256.fill(seed + 3);
  return a;
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId d;
  d.backend = hal::BackendId::cuda;
  d.api_family = hal::ApiFamily::cuda_driver;
  d.physical_device.vendor_id = 0x10de;
  d.physical_device.device_id = 1;
  d.physical_device.stable_id[0] = 1;
  d.logical_context_generation = 7;
  d.allocator_family = hal::AllocatorFamily::cuda_driver;
  d.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  d.queue_or_stream_identity = 99;
  d.ownership_model = hal::OwnershipModel::adapter_exclusive;
  d.synchronization_protocol = hal::SynchronizationProtocol::cuda_stream_ordered;
  d.adapter_instance_id = 8;
  return d;
}

PreparedCudaFastDvdnetConfig Config(ModelPack *pack) {
  PreparedCudaFastDvdnetConfig c;
  c.model_pack = pack; c.artifact = Artifact(); c.execution_domain = Domain();
  c.configuration_generation = 3; c.model_generation = 4;
  c.source_width = 5; c.source_height = 6;
  c.tensors = {"noisy", "noise_map", "denoised", {1,15,8,8}, {1,1,8,8},
               {1,3,8,8}, true, true};
  c.available_budget = {8192, 2048, 0, 0, 16, 1};
  c.ort_evidence.session_generation = 5;
  auto &p = c.ort_evidence.provider;
  p.provider = onnx::OrtExecutionProviderId::cuda; p.cuda_device_id = 0;
  p.cuda_context_generation = 7; p.requested_stream_identity = 99;
  p.user_stream_bound = true; p.cpu_ep_fallback_disabled = true;
  return c;
}

cuda::CudaImage Image(cuda::CUdeviceptr ptr) {
  cuda::CudaImage i; i.ptr=ptr; i.pitch=15; i.w=5; i.h=6;
  i.format=cuda::PixelFormatGpu::rgb_u8; return i;
}
} // namespace

int main() {
  bool ok = true;
  ModelPack pack; pack.task="video_denoise"; pack.id="fastdvdnet_sigma15";
  FakeRuntime runtime;
  PreparedCudaFastDvdnetDenoiser denoiser(&runtime);
  auto config=Config(&pack);
  auto setup=denoiser.Prepare(config);
  ok = Check(setup.status==PreparedCudaFastDvdnetStatus::ready,
             "exact strict CUDA contract should prepare") && ok;
  auto input=Image(100), output=Image(200);
  auto run=denoiser.Execute(setup.handle, config.artifact, config.execution_domain,
                            10, input, &output, 100,
                            hal::FailureAction::fail_open_preserve_input);
  ok = Check(run.status==PreparedCudaFastDvdnetStatus::executed && run.temporal_reset &&
             runtime.last.source_slots==std::array<std::uint8_t,5>{2,2,2,2,2} &&
             std::fabs(runtime.sigma-55.0f/255.0f)<1e-6f,
             "initial frame should fill all temporal positions and clamp noise") && ok;
  run=denoiser.Execute(setup.handle, config.artifact, config.execution_domain, 11,
                       input, &output, -1,
                       hal::FailureAction::fail_open_preserve_input);
  ok = Check(!run.temporal_reset && runtime.last.source_slots==std::array<std::uint8_t,5>{0,1,2,2,2} && runtime.sigma==0,
             "contiguous frame should preserve causal five-frame order") && ok;
  run=denoiser.Execute(setup.handle, config.artifact, config.execution_domain, 14,
                       input, &output, 50,
                       hal::FailureAction::fail_open_preserve_input);
  ok = Check(run.temporal_reset, "capture discontinuity should reset history") && ok;
  runtime.frame.status=PreparedCudaFastDvdnetStatus::inference_failed;
  runtime.frame.output_valid=false;
  run=denoiser.Execute(setup.handle, config.artifact, config.execution_domain, 15,
                       input, &output, 50,
                       hal::FailureAction::fail_open_preserve_input);
  ok = Check(run.status==PreparedCudaFastDvdnetStatus::optional_failed_open &&
             run.valid_input_preserved,
             "optional inference failure should preserve resident input") && ok;
  ok = Check(runtime.prepares==1 && runtime.executes==4,
             "execute must reuse setup and binding") && ok;
  runtime.first_shutdown_pending=true;
  auto stop=denoiser.Shutdown(setup.handle);
  ok = Check(!stop.quiescent && stop.status==PreparedCudaFastDvdnetStatus::completion_pending,
             "shutdown should retain resources while completion is pending") && ok;
  stop=denoiser.Shutdown(setup.handle);
  ok = Check(stop.quiescent && stop.status==PreparedCudaFastDvdnetStatus::shutdown,
             "shutdown should be retryable") && ok;

  FakeRuntime rejected_runtime;
  PreparedCudaFastDvdnetDenoiser rejected(&rejected_runtime);
  config=Config(&pack); config.ort_evidence.provider.cpu_ep_fallback_disabled=false;
  ok = Check(rejected.Prepare(config).status==PreparedCudaFastDvdnetStatus::wrong_provider && rejected_runtime.prepares==0,
             "CPU fallback evidence must fail before runtime setup") && ok;
  return ok ? 0 : 1;
}
