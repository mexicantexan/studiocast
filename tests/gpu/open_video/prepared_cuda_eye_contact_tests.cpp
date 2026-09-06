#include "core/open_video/native_prepared_cuda_eye_contact_runtime.h"
#include "core/open_video/prepared_cuda_eye_contact.h"

#include <dlib/image_processing/shape_predictor.h>
#include <dlib/serialize.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

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
  int *destructions = nullptr;
  ~Analysis() override {
    if (destructions)
      ++*destructions;
  }
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
  std::size_t last_budget = 0;
  int *destructions = nullptr;
  ~Runtime() override {
    if (destructions)
      ++*destructions;
  }
  PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactRuntimeConfig &config) noexcept override {
    ++prepare;
    last_budget = config.resource_budget_bytes;
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

struct NativeState {
  cuda::CUcontext current = nullptr;
  std::uint64_t next = 0x1000;
  int runs = 0, records = 0, queries = 0, frees = 0, destroys = 0;
  int fail_run_at = 0, not_ready = 0, fail_free_at = 0;
  std::size_t pitch_extra = 0;
  int yunet_runs = 0, yunet_invalidates = 0;
} * native;
cuda::CUresult CInit(unsigned) { return 0; }
cuda::CUresult CLoad(cuda::CUmodule *m, const void *) {
  *m = reinterpret_cast<cuda::CUmodule>(1);
  return 0;
}
cuda::CUresult CFunction(cuda::CUfunction *f, cuda::CUmodule, const char *) {
  *f = reinterpret_cast<cuda::CUfunction>(2);
  return 0;
}
cuda::CUresult CUnload(cuda::CUmodule) { return 0; }
cuda::CUresult CLaunch(cuda::CUfunction, unsigned, unsigned, unsigned, unsigned,
                       unsigned, unsigned, unsigned, cuda::CUstream, void **,
                       void **) {
  return 0;
}
cuda::CUresult CAlloc(cuda::CUdeviceptr *p, std::size_t *pitch, std::size_t w,
                      std::size_t h, unsigned) {
  *pitch = w == 8192 ? w : w + native->pitch_extra;
  *p = reinterpret_cast<cuda::CUdeviceptr>(std::calloc(1, *pitch * h));
  if (!*p)
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
  return 0;
}
cuda::CUresult CFree(cuda::CUdeviceptr p) {
  ++native->frees;
  if (native->fail_free_at == native->frees)
    return cuda::CUDA_ERROR_INVALID_CONTEXT;
  std::free(reinterpret_cast<void *>(static_cast<std::uintptr_t>(p)));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CCopy(const cuda::CUDA_MEMCPY2D *, cuda::CUstream) { return 0; }
cuda::CUresult CMemset(cuda::CUdeviceptr, unsigned char, std::size_t,
                       cuda::CUstream) {
  return 0;
}
cuda::CUresult CHostAlloc(void **p, std::size_t n, unsigned) {
  *p = std::calloc(1, n);
  return *p ? cuda::CUDA_SUCCESS : cuda::CUDA_ERROR_OUT_OF_MEMORY;
}
cuda::CUresult CHostFree(void *p) {
  std::free(p);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult CStream(cuda::CUstream *s, unsigned) {
  *s = reinterpret_cast<cuda::CUstream>(9);
  return 0;
}
cuda::CUresult CDestroyStream(cuda::CUstream) { return 0; }
cuda::CUresult CSync(cuda::CUstream) { return 0; }
cuda::CUresult CCount(int *count) {
  *count = 1;
  return 0;
}
cuda::CUresult CDevice(cuda::CUdevice *device, int) {
  *device = 0;
  return 0;
}
cuda::CUresult CUuid(cuda::CUuuid *uuid, cuda::CUdevice) {
  std::memset(uuid, 0, sizeof(*uuid));
  uuid->bytes[0] = 1;
  return 0;
}
cuda::CUresult CPci(char *text, int size, cuda::CUdevice) {
  const char value[] = "0000:01:00.0";
  if (size < static_cast<int>(sizeof(value)))
    return 1;
  std::memcpy(text, value, sizeof(value));
  return 0;
}
cuda::CUresult CRetain(cuda::CUcontext *context, cuda::CUdevice) {
  *context = reinterpret_cast<cuda::CUcontext>(7);
  return 0;
}
cuda::CUresult CRelease(cuda::CUdevice) { return 0; }
cuda::CUresult CSet(cuda::CUcontext context) {
  native->current = context;
  return 0;
}
cuda::CUresult CGet(cuda::CUcontext *context) {
  *context = native->current;
  return 0;
}
cuda::CUresult CGetDevice(cuda::CUdevice *device) {
  *device = 0;
  return 0;
}
cuda::CUresult CEventCreate(cuda::CUevent *event, unsigned) {
  *event = reinterpret_cast<cuda::CUevent>(8);
  return 0;
}
cuda::CUresult CEventDestroy(cuda::CUevent) {
  ++native->destroys;
  return 0;
}
cuda::CUresult CEventRecord(cuda::CUevent, cuda::CUstream) {
  ++native->records;
  return 0;
}
cuda::CUresult CEventQuery(cuda::CUevent) {
  ++native->queries;
  if (native->not_ready > 0) {
    --native->not_ready;
    return cuda::CUDA_ERROR_NOT_READY;
  }
  return 0;
}
cuda::CUresult CElapsed(float *ms, cuda::CUevent, cuda::CUevent) {
  *ms = 0;
  return 0;
}
cuda::CudaDriverApi::Functions CudaFunctions() {
  cuda::CudaDriverApi::Functions f;
  f.cuInit = CInit;
  f.cuModuleLoadData = CLoad;
  f.cuModuleGetFunction = CFunction;
  f.cuModuleUnload = CUnload;
  f.cuLaunchKernel = CLaunch;
  f.cuMemAllocPitch = CAlloc;
  f.cuMemFree = CFree;
  f.cuMemcpy2DAsync = CCopy;
  f.cuMemsetD8Async = CMemset;
  f.cuMemHostAlloc = CHostAlloc;
  f.cuMemFreeHost = CHostFree;
  f.cuStreamCreate = CStream;
  f.cuStreamDestroy = CDestroyStream;
  f.cuStreamSynchronize = CSync;
  f.cuDeviceGetCount = CCount;
  f.cuDeviceGet = CDevice;
  f.cuDeviceGetUuid = CUuid;
  f.cuDeviceGetPCIBusId = CPci;
  f.cuDevicePrimaryCtxRetain = CRetain;
  f.cuDevicePrimaryCtxRelease = CRelease;
  f.cuCtxSetCurrent = CSet;
  f.cuCtxGetCurrent = CGet;
  f.cuCtxGetDevice = CGetDevice;
  f.cuEventCreate = CEventCreate;
  f.cuEventDestroy = CEventDestroy;
  f.cuEventRecord = CEventRecord;
  f.cuEventQuery = CEventQuery;
  f.cuEventElapsedTime = CElapsed;
  return f;
}
struct YunetSession final : internal::IPreparedCudaYunetSession {
  onnx::OrtSessionInfo info_;
  YunetSession() {
    info_.strict_provider_validation_passed = true;
    info_.input_names = {"input"};
    info_.input_shapes = {{1, 3, 320, 320}};
    info_.input_elem_types = {1};
    for (auto name : {"cls_8", "cls_16", "cls_32", "obj_8", "obj_16",
                      "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8",
                      "kps_16", "kps_32"})
      info_.output_names.emplace_back(name);
    for (int index = 0; index < 12; ++index) {
      const int stride = index % 3 == 0 ? 8 : (index % 3 == 1 ? 16 : 32);
      const int channels = index < 6 ? 1 : (index < 9 ? 4 : 10);
      info_.output_shapes.push_back(
          {1, (320 / stride) * (320 / stride), channels});
    }
    auto &provider = info_.prepared_provider_evidence;
    provider.provider = onnx::OrtExecutionProviderId::cuda;
    provider.cuda_device_id = 0;
    provider.cuda_context_generation = 1;
    provider.requested_stream_identity = 9;
    provider.user_stream_bound = true;
    provider.cpu_ep_fallback_disabled = true;
  }
  const onnx::OrtSessionInfo &info() const noexcept override { return info_; }
  bool Prepare(const onnx::OrtSession::CudaBindingInput *, std::size_t ni,
               const onnx::OrtSession::CudaBindingOutput *, std::size_t no,
               std::string *) override {
    return ni == 1 && no == 12;
  }
  onnx::OrtPreparedCudaRunEvidence evidence() const noexcept override {
    onnx::OrtPreparedCudaRunEvidence evidence;
    evidence.provider = info_.prepared_provider_evidence;
    evidence.session_generation = 1;
    return evidence;
  }
  onnx::OrtPreparedRunResult Execute(
      const onnx::OrtSession::CudaBindingInput *, std::size_t,
      const onnx::OrtSession::CudaBindingOutput *outputs, std::size_t count,
      const onnx::OrtPreparedCudaRunEvidence &) noexcept override {
    ++native->yunet_runs;
    for (std::size_t i = 0; i < count; ++i)
      std::memset(outputs[i].device_ptr, 0,
                  outputs[i].num_floats * sizeof(float));
    onnx::OrtPreparedRunResult result;
    result.status = onnx::OrtPreparedRunStatus::success;
    return result;
  }
  void Invalidate() noexcept override { ++native->yunet_invalidates; }
};

std::filesystem::path TinyDlibPack() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("studiocast-eye-shared-dlib-" +
                     std::to_string(::getpid()));
  const auto dir = root / "face_landmarks" / "tiny-dlib68";
  std::filesystem::create_directories(dir);
  dlib::matrix<float, 0, 1> shape(136);
  for (long i = 0; i < 68; ++i) {
    shape(2 * i) = 0.2F + 0.005F * static_cast<float>(i);
    shape(2 * i + 1) = 0.3F + 0.004F * static_cast<float>(i);
  }
  dlib::impl::regression_tree tree;
  tree.splits.push_back({0, 1, 0.0F});
  tree.leaf_values.emplace_back(136);
  tree.leaf_values.emplace_back(136);
  tree.leaf_values[0] = 0;
  tree.leaf_values[1] = 0;
  std::vector<std::vector<dlib::impl::regression_tree>> forests{{tree}};
  std::vector<std::vector<dlib::vector<float, 2>>> pixels{
      {{0.5F, 0.5F}, {0.25F, 0.25F}}};
  dlib::serialize((dir / "model.dat").string())
      << dlib::shape_predictor(shape, forests, pixels);
  std::ofstream(dir / "LICENSE.txt") << "synthetic test fixture\n";
  std::ofstream(dir / "model.json")
      << R"({"schema_version":2,"id":"tiny-dlib68","display_name":"tiny","task":"face_landmarks","files":[{"name":"model.dat","kind":"dlib_shape_predictor","role":"main","sha256":""}],"dlib":{"predictor_kind":"shape_predictor_68_face_landmarks","num_landmarks":68}})";
  return root;
}
struct EyeSession final : IPreparedEyeContactOrtSession {
  onnx::OrtSessionInfo metadata;
  onnx::OrtPreparedCudaRunEvidence run_evidence;
  int *run_count = nullptr;
  int *destruction_count = nullptr;
  ~EyeSession() override {
    if (destruction_count)
      ++*destruction_count;
  }
  const onnx::OrtSessionInfo &info() const noexcept override {
    return metadata;
  }
  onnx::OrtPreparedCudaRunEvidence evidence() const noexcept override {
    return run_evidence;
  }
  bool PrepareBinding(const onnx::OrtSession::CudaBindingInput *in,
                      std::size_t n,
                      const onnx::OrtSession::CudaBindingOutput *out,
                      std::size_t no) noexcept override {
    return n == 3 && no == 1 && std::strcmp(in[0].name, "eye_rgb:0") == 0 &&
           std::strcmp(out[0].name, "rgb:0") == 0;
  }
  onnx::OrtPreparedRunResult
  Execute(const onnx::OrtSession::CudaBindingInput *, std::size_t,
          const onnx::OrtSession::CudaBindingOutput *, std::size_t,
          const onnx::OrtPreparedCudaRunEvidence &) noexcept override {
    ++*run_count;
    onnx::OrtPreparedRunResult result;
    result.status = native->fail_run_at == *run_count
                        ? onnx::OrtPreparedRunStatus::inference_failure
                        : onnx::OrtPreparedRunStatus::success;
    return result;
  }
  void RequestShutdown() noexcept override {}
};
struct EyeFactory final : IPreparedEyeContactOrtFactory {
  onnx::OrtSessionInfo metadata;
  onnx::OrtPreparedCudaRunEvidence evidence;
  int runs = 0, validations = 0;
  int session_destructions = 0;
  bool validate = true;
  bool
  ValidateArtifact(const ModelPack &, const ModelFile &,
                   const ValidatedModelArtifactIdentity &) noexcept override {
    ++validations;
    return validate;
  }
  std::unique_ptr<IPreparedEyeContactOrtSession>
  Create(const std::filesystem::path &,
         const onnx::OrtSessionOptions &options) noexcept override {
    if (options.provider_policy != onnx::OrtProviderPolicy::strict_cuda_only ||
        options.enable_tensorrt || !options.user_compute_stream)
      return {};
    auto session = std::make_unique<EyeSession>();
    session->metadata = metadata;
    session->run_evidence = evidence;
    session->run_count = &runs;
    session->destruction_count = &session_destructions;
    return session;
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
  s.yunet_setup.resource_budget_bytes = 1000;
  s.dlib_setup.resource_budget_bytes = 1000;
  s.wait.timeout_nanoseconds = 1000;
  s.wait.poll_interval_nanoseconds = 1;
  s.wait.maximum_poll_count = 10;
  static ModelPack pack;
  static ModelFile left, right;
  left.role = "left";
  left.path = s.models.left_path;
  right.role = "right";
  right.path = s.models.right_path;
  s.models.eye_pack = &pack;
  s.models.left_file = &left;
  s.models.right_file = &right;
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
  cuda::CudaTensor resident;
  resident.ptr = 1;
  resident.pitch = resident.bytes = 136 * sizeof(float);
  resident.n = resident.c = resident.h = 1;
  resident.w = 136;
  PreparedCudaDlibResult resident_result;
  resident_result.status = PreparedCudaDlibStatus::success;
  resident_result.pending_work = true;
  CHECK(ValidateResidentDlibLandmarks(resident_result, resident) ==
        PreparedCudaEyeContactStatus::executed);
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
  runtime.next.completion_waits = 2;
  r = eye.Execute(h, setup.binding, 3, input);
  CHECK(r.status == PreparedCudaEyeContactStatus::inference_failed &&
        r.valid_input_preserved && r.output == &input &&
        r.completion_waits == 3);
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
  PreparedCudaEyeContactStatus prepare_status;
  CHECK(!eye.Prepare(Setup(&cache), &h, &error, &prepare_status));
  CHECK(prepare_status == PreparedCudaEyeContactStatus::provider_mismatch);
  CHECK(!eye.ready());
  runtime.prepare_status = PreparedCudaEyeContactStatus::graph_mismatch;
  CHECK(!eye.Prepare(Setup(&cache), &h, &error, &prepare_status));
  CHECK(prepare_status == PreparedCudaEyeContactStatus::graph_mismatch);
  setup = Setup(&cache);
  setup.resource_budget_bytes = setup.yunet_setup.resource_budget_bytes +
                                setup.dlib_setup.resource_budget_bytes - 1;
  CHECK(!eye.Prepare(setup, &h, &error, &prepare_status));
  CHECK(prepare_status == PreparedCudaEyeContactStatus::resource_exhausted);
  int analysis_destructions = 0, runtime_destructions = 0;
  {
    auto owned_analysis = std::make_unique<Analysis>();
    auto owned_runtime = std::make_unique<Runtime>();
    owned_analysis->destructions = &analysis_destructions;
    owned_runtime->destructions = &runtime_destructions;
    owned_runtime->shutdown_status =
        PreparedCudaEyeContactStatus::shutdown_failed;
    PreparedCudaEyeContact owned(std::move(owned_analysis),
                                 std::move(owned_runtime));
    CHECK(owned.Prepare(Setup(&cache), &h, &error));
  }
  CHECK(analysis_destructions == 0 && runtime_destructions == 0);
  {
    auto owned_analysis = std::make_unique<Analysis>();
    auto owned_runtime = std::make_unique<Runtime>();
    owned_analysis->destructions = &analysis_destructions;
    owned_runtime->destructions = &runtime_destructions;
    PreparedCudaEyeContact owned(std::move(owned_analysis),
                                 std::move(owned_runtime));
    CHECK(owned.Prepare(Setup(&cache), &h, &error));
  }
  CHECK(analysis_destructions == 1 && runtime_destructions == 1);

  // A shared YuNet is budgeted by the outer slot. Eye accounts only for its
  // owned dlib analysis and eye runtime reserve.
  Analysis shared_analysis;
  Runtime shared_runtime;
  PreparedCudaEyeContact shared_eye(&shared_analysis, &shared_runtime);
  setup = Setup(&cache);
  PreparedCudaYunetPreparedBinding borrowed;
  setup.shared_yunet = &borrowed;
  setup.resource_budget_bytes = setup.dlib_setup.resource_budget_bytes + 123;
  CHECK(shared_eye.Prepare(setup, &h, &error));
  CHECK(shared_runtime.last_budget == 123);
  CHECK(shared_eye.Shutdown(&error));
  setup.resource_budget_bytes = setup.dlib_setup.resource_budget_bytes - 1;
  CHECK(!shared_eye.Prepare(setup, &h, &error));

  Analysis retry_analysis;
  Runtime retry_runtime;
  PreparedCudaEyeContact retry_eye(&retry_analysis, &retry_runtime);
  setup = Setup(&cache);
  CHECK(retry_eye.Prepare(setup, &h, &error));
  retry_runtime.shutdown_status = PreparedCudaEyeContactStatus::shutdown_failed;
  CHECK(!retry_eye.Prepare(setup, &h, &error, &prepare_status));
  CHECK(prepare_status == PreparedCudaEyeContactStatus::shutdown_failed);
  retry_runtime.shutdown_status = PreparedCudaEyeContactStatus::shutdown;
  CHECK(retry_eye.Prepare(setup, &h, &error, &prepare_status));
  CHECK(prepare_status == PreparedCudaEyeContactStatus::executed);
  CHECK(retry_eye.Shutdown(&error));
  return true;
}
bool ConcreteSharedAnalysis() {
  NativeState state;
  native = &state;
  cuda::CudaDriverApi driver;
  std::string error;
  CHECK(driver.InitializeForTesting(CudaFunctions(), &error));
  CHECK(driver.EnsureContextForDevice(0, &error));
  const auto dlib_root = TinyDlibPack();
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{dlib_root};
  auto registry = ModelPackRegistry::Scan(dlib_root);
  auto dlib_pack = registry.Find("face_landmarks", "tiny-dlib68");
  CHECK(dlib_pack && !dlib_pack->files.empty());
  const auto dlib_identity =
      ValidateModelArtifactIdentity(*dlib_pack, dlib_pack->files.front());
  CHECK(dlib_identity.ok());

  PreparedCudaYunetSetup yunet_setup;
  yunet_setup.binding = {&driver, *driver.context_identity(),
                         reinterpret_cast<cuda::CUstream>(9), 9, 7, 8};
  yunet_setup.frame_width = 64;
  yunet_setup.frame_height = 48;
  yunet_setup.resource_budget_bytes =
      PreparedCudaYunetDetector::RequiredResourceBytes();
  yunet_setup.declared_input = {"input", "nchw", "float32", 320, 320, 3};
  const auto yunet_identity = Identity(11);
  PreparedCudaYunetDetector yunet;
  CHECK(yunet.PrepareForTesting(yunet_setup, yunet_identity,
                                std::make_unique<YunetSession>(), &error));
  FrameAnalysisCache cache;
  CHECK(yunet.PrepareCache(&cache, &error));
  PreparedCudaYunetPreparedBinding shared;
  CHECK(yunet.ExportPreparedBinding(&cache, &shared));
  const int runs_after_warmup = state.yunet_runs;

  Runtime runtime;
  PreparedCudaEyeContact eye(&runtime);
  auto setup = Setup(&cache);
  setup.binding = {&driver,
                   *driver.context_identity(),
                   state.current,
                   reinterpret_cast<cuda::CUstream>(9),
                   9,
                   7,
                   8,
                   9};
  setup.frame_width = 64;
  setup.frame_height = 48;
  setup.models.yunet = yunet_identity;
  setup.models.landmarks = dlib_identity.identity;
  ++setup.models.landmarks.registry_generation;
  setup.shared_yunet = &shared;
  setup.dlib_setup.models_root = dlib_root;
  setup.dlib_setup.model_id = "tiny-dlib68";
  setup.dlib_setup.resource_budget_bytes = 1U << 20;
  setup.resource_budget_bytes = setup.dlib_setup.resource_budget_bytes + 4096;
  PreparedCudaEyeContactHandle handle;
  if (!eye.Prepare(setup, &handle, &error)) {
    std::cerr << "concrete shared prepare: " << error << '\n';
    return false;
  }
  CHECK(state.yunet_runs == runs_after_warmup);
  cuda::CudaImage input;
  input.ptr = 1;
  input.pitch = 192;
  input.w = 64;
  input.h = 48;
  input.format = cuda::PixelFormatGpu::rgb_u8;
  auto result = eye.Execute(handle, setup.binding, 50, input);
  CHECK(result.status == PreparedCudaEyeContactStatus::preserved_input &&
        !result.face_found && state.yunet_runs == runs_after_warmup + 1);
  const auto direct = yunet.Execute(yunet_setup.binding, input, 50, &cache,
                                    setup.wait);
  CHECK(direct.status == PreparedCudaYunetStatus::success_no_faces &&
        direct.enqueued_readbacks == 0 && direct.completion_waits == 0 &&
        state.yunet_runs == runs_after_warmup + 1);
  CHECK(eye.Shutdown(&error));
  PreparedCudaEyeContactStatus budget_status;
  Runtime budget_runtime;
  PreparedCudaEyeContact budget_eye(&budget_runtime);
  state.pitch_extra = 1U << 20;
  CHECK(!budget_eye.Prepare(setup, &handle, &error, &budget_status));
  CHECK(budget_status == PreparedCudaEyeContactStatus::resource_exhausted);
  state.pitch_extra = 0;
  CHECK(yunet.ready() && state.yunet_invalidates == 0);
  CHECK(yunet.Shutdown(&error) && state.yunet_invalidates == 1);
  CHECK(yunet.PrepareForTesting(yunet_setup, yunet_identity,
                                std::make_unique<YunetSession>(), &error));
  CHECK(yunet.PrepareCache(&cache, &error));
  Runtime stale_runtime;
  PreparedCudaEyeContact stale_eye(&stale_runtime);
  PreparedCudaEyeContactStatus stale_status;
  CHECK(!stale_eye.Prepare(setup, &handle, &error, &stale_status));
  CHECK(stale_status == PreparedCudaEyeContactStatus::model_invalid &&
        stale_runtime.prepare == 0);
  CHECK(yunet.Shutdown(&error));
  return true;
}
bool NativeBoundaries() {
  NativeState state;
  native = &state;
  cuda::CudaDriverApi driver;
  std::string error;
  CHECK(driver.InitializeForTesting(CudaFunctions(), &error));
  CHECK(driver.EnsureContextForDevice(0, &error));
  PreparedCudaEyeContactRuntimeConfig config;
  config.binding.driver = &driver;
  config.binding.context = *driver.context_identity();
  config.binding.owner_context = state.current;
  config.binding.stream = reinterpret_cast<cuda::CUstream>(9);
  config.binding.stream_identity = 9;
  config.binding.owner_identity = 7;
  config.binding.configuration_generation = 8;
  config.binding.model_generation = 9;
  config.cuda_device_id = 0;
  config.frame_width = 64;
  config.frame_height = 48;
  config.resource_budget_bytes = 1 << 20;
  config.wait.timeout_nanoseconds = 100;
  config.wait.poll_interval_nanoseconds = 1;
  config.wait.maximum_poll_count = 2;
  ModelPack pack;
  ModelFile left, right;
  left.role = "left";
  left.path = "left.onnx";
  right.role = "right";
  right.path = "right.onnx";
  pack.files = {left, right};
  config.models.eye_pack = &pack;
  config.models.left_file = &pack.files[0];
  config.models.right_file = &pack.files[1];
  config.models.left_path = left.path;
  config.models.right_path = right.path;
  config.models.left = Identity(3);
  config.models.right = Identity(4);
  EyeFactory factory;
  factory.metadata.input_names = {"eye_rgb:0", "anchors:0", "angles:0"};
  factory.metadata.output_names = {"rgb:0"};
  factory.metadata.input_shapes = {{1, 3, 48, 64}, {1, 12, 48, 64}, {1, 2}};
  factory.metadata.output_shapes = {{1, 3, 48, 64}};
  factory.metadata.input_elem_types = {1, 1, 1};
  factory.metadata.output_elem_types = {1};
  factory.evidence.session_generation = 1;
  auto &provider = factory.evidence.provider;
  provider.provider = onnx::OrtExecutionProviderId::cuda;
  provider.cuda_device_id = 0;
  provider.cuda_context_generation = config.binding.context.logical_generation;
  provider.requested_stream_identity =
      reinterpret_cast<std::uintptr_t>(config.binding.stream);
  provider.user_stream_bound = true;
  provider.cpu_ep_fallback_disabled = true;
  NativePreparedCudaEyeContactRuntime runtime(&factory);
  CHECK(runtime.Prepare(config) == PreparedCudaEyeContactStatus::executed);
  CHECK(factory.validations == 2 && factory.runs == 2 && state.records == 1);
  cuda::CudaImage input;
  input.ptr = 0x2000000;
  input.pitch = 192;
  input.w = 64;
  input.h = 48;
  input.format = cuda::PixelFormatGpu::rgb_u8;
  cuda::CudaTensor landmarks;
  landmarks.ptr = 0x3000000;
  landmarks.pitch = landmarks.bytes = 136 * sizeof(float);
  landmarks.n = 1;
  landmarks.c = 1;
  landmarks.h = 1;
  landmarks.w = 136;
  state.fail_run_at = 4;
  auto result = runtime.Execute(config.binding, input, landmarks, 0.5F);
  CHECK(result.status == PreparedCudaEyeContactStatus::inference_failed &&
        result.output == &input && !result.pending_work && factory.runs == 4 &&
        result.completion_waits == 1);
  CHECK(runtime.Shutdown() == PreparedCudaEyeContactStatus::shutdown);
  CHECK(factory.session_destructions == 2 && state.frees == 10);

  state = {};
  state.next = 0x1000;
  native = &state;
  CHECK(driver.BindPreparedContext(config.binding.context) ==
        cuda::CUDA_SUCCESS);
  EyeFactory timeout_factory = factory;
  timeout_factory.runs = 0;
  NativePreparedCudaEyeContactRuntime timeout_runtime(&timeout_factory);
  CHECK(timeout_runtime.Prepare(config) ==
        PreparedCudaEyeContactStatus::executed);
  state.not_ready = 2;
  state.fail_run_at = 4;
  result = timeout_runtime.Execute(config.binding, input, landmarks, 0.5F);
  CHECK(result.status == PreparedCudaEyeContactStatus::timeout &&
        result.pending_work && result.output == nullptr &&
        result.completion_waits == 1);
  CHECK(timeout_runtime.Shutdown() ==
        PreparedCudaEyeContactStatus::shutdown_failed);
  CHECK(timeout_runtime.Shutdown() == PreparedCudaEyeContactStatus::shutdown);

  state = {};
  state.next = 0x1000;
  native = &state;
  EyeFactory destructor_factory = factory;
  destructor_factory.runs = 0;
  destructor_factory.session_destructions = 0;
  {
    NativePreparedCudaEyeContactRuntime pending(&destructor_factory);
    CHECK(pending.Prepare(config) == PreparedCudaEyeContactStatus::executed);
    state.not_ready = 100;
    state.fail_run_at = 4;
    result = pending.Execute(config.binding, input, landmarks, 0.5F);
    CHECK(result.status == PreparedCudaEyeContactStatus::timeout &&
          result.pending_work);
  }
  CHECK(destructor_factory.session_destructions == 0 && state.frees == 0 &&
        state.destroys == 0);

  state = {};
  state.next = 0x1000;
  native = &state;
  EyeFactory cleanup_factory = factory;
  cleanup_factory.runs = 0;
  NativePreparedCudaEyeContactRuntime cleanup(&cleanup_factory);
  CHECK(cleanup.Prepare(config) == PreparedCudaEyeContactStatus::executed);
  state.fail_free_at = 1;
  CHECK(cleanup.Shutdown() == PreparedCudaEyeContactStatus::shutdown_failed);
  state.fail_free_at = 0;
  CHECK(cleanup.Shutdown() == PreparedCudaEyeContactStatus::shutdown);

  EyeFactory rejected = factory;
  rejected.validations = 0;
  rejected.validate = false;
  NativePreparedCudaEyeContactRuntime invalid(&rejected);
  CHECK(invalid.Prepare(config) ==
            PreparedCudaEyeContactStatus::invalid_setup &&
        rejected.validations == 1);
  config.resource_budget_bytes = 1;
  EyeFactory budget_factory = factory;
  NativePreparedCudaEyeContactRuntime budget(&budget_factory);
  CHECK(budget.Prepare(config) ==
        PreparedCudaEyeContactStatus::resource_exhausted);
  return true;
}
} // namespace
int main() {
  return ContractsAndBehavior() && FailureAndDisabled() &&
                 ConcreteSharedAnalysis() && NativeBoundaries()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
