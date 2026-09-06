#include "core/open_video/prepared_cuda_fastdvdnet_denoiser.h"
#include "core/open_video/native_prepared_cuda_fastdvdnet_runtime.h"

#include <cmath>
#include <iostream>
#include <cstring>

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
  bool preserve_prepare_evidence = false;

  PreparedCudaFastDvdnetRuntimePrepareResult
  Prepare(const PreparedCudaFastDvdnetConfig &c) noexcept override {
    ++prepares;
    if (!preserve_prepare_evidence) {
      prepare.ort_evidence.session_generation = 5;
      auto &p = prepare.ort_evidence.provider;
      p.provider = onnx::OrtExecutionProviderId::cuda; p.cuda_device_id = 0;
      p.cuda_context_generation = c.execution_domain.logical_context_generation;
      p.requested_stream_identity = c.execution_domain.queue_or_stream_identity;
      p.user_stream_bound = true; p.cpu_ep_fallback_disabled = true;
    }
    return prepare;
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
  return c;
}

onnx::OrtPreparedCudaRunEvidence Evidence(const PreparedCudaFastDvdnetConfig &c) {
  onnx::OrtPreparedCudaRunEvidence evidence;
  evidence.session_generation = 5;
  auto &p = evidence.provider;
  p.provider = onnx::OrtExecutionProviderId::cuda; p.cuda_device_id = 0;
  p.cuda_context_generation = c.execution_domain.logical_context_generation;
  p.requested_stream_identity = c.execution_domain.queue_or_stream_identity;
  p.user_stream_bound = true; p.cpu_ep_fallback_disabled = true;
  return evidence;
}

cuda::CudaImage Image(cuda::CUdeviceptr ptr) {
  cuda::CudaImage i; i.ptr=ptr; i.pitch=15; i.w=5; i.h=6;
  i.format=cuda::PixelFormatGpu::rgb_u8; return i;
}

struct NativeCudaState {
  cuda::CUcontext current=nullptr; std::uint64_t next=0x1000;
  int allocs=0, frees=0, successful_frees=0, unloads=0, event_destroys=0,
      launches=0, copies=0, memsets=0, records=0, syncs=0, queries=0;
  int fail_launch_at=0, fail_copy_at=0, fail_memset_at=0, fail_record_at=0,
      fail_syncs=0, not_ready_queries=0, fail_free_at=0, fail_unload_at=0,
      fail_event_destroy_at=0;
} *native_state;
cuda::CUresult NInit(unsigned){return 0;} cuda::CUresult NLoad(cuda::CUmodule*m,const void*){*m=reinterpret_cast<cuda::CUmodule>(1);return 0;}
cuda::CUresult NFn(cuda::CUfunction*f,cuda::CUmodule,const char*n){*f=reinterpret_cast<cuda::CUfunction>(std::strcmp(n,"fastdvdnet_preprocess")==0?2:3);return 0;}
cuda::CUresult NUnload(cuda::CUmodule){++native_state->unloads;return native_state->fail_unload_at==native_state->unloads?cuda::CUDA_ERROR_INVALID_CONTEXT:0;} cuda::CUresult NLaunch(cuda::CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,cuda::CUstream,void**,void**){++native_state->launches;return native_state->fail_launch_at==native_state->launches?cuda::CUDA_ERROR_LAUNCH_FAILED:0;}
cuda::CUresult NAlloc(cuda::CUdeviceptr*p,std::size_t*pi,std::size_t w,std::size_t,unsigned){*p=native_state->next;native_state->next+=0x10000;*pi=w;++native_state->allocs;return 0;}
cuda::CUresult NFree(cuda::CUdeviceptr){++native_state->frees;if(native_state->fail_free_at==native_state->frees)return cuda::CUDA_ERROR_INVALID_CONTEXT;++native_state->successful_frees;return 0;} cuda::CUresult NCopy(const cuda::CUDA_MEMCPY2D*,cuda::CUstream){++native_state->copies;return native_state->fail_copy_at==native_state->copies?cuda::CUDA_ERROR_ILLEGAL_ADDRESS:0;}
cuda::CUresult NMemset(cuda::CUdeviceptr,unsigned char,std::size_t,cuda::CUstream){++native_state->memsets;return native_state->fail_memset_at==native_state->memsets?cuda::CUDA_ERROR_LAUNCH_FAILED:0;} cuda::CUresult NStream(cuda::CUstream*s,unsigned){*s=reinterpret_cast<cuda::CUstream>(9);return 0;}
cuda::CUresult NDestroyStream(cuda::CUstream){return 0;} cuda::CUresult NSync(cuda::CUstream){++native_state->syncs;if(native_state->fail_syncs>0){--native_state->fail_syncs;return cuda::CUDA_ERROR_NOT_READY;}return 0;} cuda::CUresult NCount(int*c){*c=1;return 0;} cuda::CUresult NDev(cuda::CUdevice*d,int){*d=0;return 0;}
cuda::CUresult NUuid(cuda::CUuuid*u,cuda::CUdevice){std::memset(u,0,sizeof(*u));u->bytes[0]=1;return 0;} cuda::CUresult NPci(char*b,int n,cuda::CUdevice){const char*s="0000:01:00.0";if(n<13)return 1;std::memcpy(b,s,13);return 0;}
cuda::CUresult NRetain(cuda::CUcontext*c,cuda::CUdevice){*c=reinterpret_cast<cuda::CUcontext>(7);return 0;} cuda::CUresult NRelease(cuda::CUdevice){return 0;}
cuda::CUresult NSet(cuda::CUcontext c){native_state->current=c;return 0;} cuda::CUresult NGet(cuda::CUcontext*c){*c=native_state->current;return 0;} cuda::CUresult NGetDev(cuda::CUdevice*d){*d=0;return 0;}
cuda::CUresult NECreate(cuda::CUevent*e,unsigned){*e=reinterpret_cast<cuda::CUevent>(8);return 0;} cuda::CUresult NEDestroy(cuda::CUevent){++native_state->event_destroys;return native_state->fail_event_destroy_at==native_state->event_destroys?cuda::CUDA_ERROR_INVALID_CONTEXT:0;}
cuda::CUresult NERecord(cuda::CUevent,cuda::CUstream){++native_state->records;return native_state->fail_record_at==native_state->records?cuda::CUDA_ERROR_LAUNCH_FAILED:0;} cuda::CUresult NEQuery(cuda::CUevent){++native_state->queries;if(native_state->not_ready_queries>0){--native_state->not_ready_queries;return cuda::CUDA_ERROR_NOT_READY;}return 0;} cuda::CUresult NEElapsed(float*m,cuda::CUevent,cuda::CUevent){*m=0;return 0;}
cuda::CudaDriverApi::Functions NativeFunctions(){cuda::CudaDriverApi::Functions f;f.cuInit=NInit;f.cuModuleLoadData=NLoad;f.cuModuleGetFunction=NFn;f.cuModuleUnload=NUnload;f.cuLaunchKernel=NLaunch;f.cuMemAllocPitch=NAlloc;f.cuMemFree=NFree;f.cuMemcpy2DAsync=NCopy;f.cuMemsetD8Async=NMemset;f.cuStreamCreate=NStream;f.cuStreamDestroy=NDestroyStream;f.cuStreamSynchronize=NSync;f.cuDeviceGetCount=NCount;f.cuDeviceGet=NDev;f.cuDeviceGetUuid=NUuid;f.cuDeviceGetPCIBusId=NPci;f.cuDevicePrimaryCtxRetain=NRetain;f.cuDevicePrimaryCtxRelease=NRelease;f.cuCtxSetCurrent=NSet;f.cuCtxGetCurrent=NGet;f.cuCtxGetDevice=NGetDev;f.cuEventCreate=NECreate;f.cuEventDestroy=NEDestroy;f.cuEventRecord=NERecord;f.cuEventQuery=NEQuery;f.cuEventElapsedTime=NEElapsed;return f;}
struct FakeOrt final:IPreparedFastDvdnetOrtSession{onnx::OrtPreparedCudaRunEvidence ev;int binds=0,runs=0,fail_run_at=0;bool binding_valid=true;explicit FakeOrt(onnx::OrtPreparedCudaRunEvidence e,int fail):ev(e),fail_run_at(fail){} auto evidence()const noexcept->onnx::OrtPreparedCudaRunEvidence override{return ev;} bool PrepareBinding(const onnx::OrtSession::CudaBindingInput*i,std::size_t,const onnx::OrtSession::CudaBindingOutput*o,std::size_t)noexcept override{++binds;binding_valid=std::strcmp(i[0].name,"noisy")==0&&std::strcmp(i[1].name,"noise_map")==0&&std::strcmp(o[0].name,"denoised")==0&&i[0].shape[1]==15&&o[0].shape[1]==3;return binding_valid;} onnx::OrtPreparedRunResult Execute(const onnx::OrtSession::CudaBindingInput*i,std::size_t,const onnx::OrtSession::CudaBindingOutput*o,std::size_t,const onnx::OrtPreparedCudaRunEvidence&)noexcept override{++runs;binding_valid=binding_valid&&std::strcmp(i[0].name,"noisy")==0&&i[0].shape[1]==15&&o[0].shape[1]==3;onnx::OrtPreparedRunResult r;r.status=(fail_run_at==runs)?onnx::OrtPreparedRunStatus::inference_failure:onnx::OrtPreparedRunStatus::success;return r;}void RequestShutdown()noexcept override{}};
struct FakeOrtFactory final:IPreparedFastDvdnetOrtFactory{onnx::OrtPreparedCudaRunEvidence ev;FakeOrt*made=nullptr;int fail_run_at=0;auto Create(const std::filesystem::path&,const onnx::OrtSessionOptions&o,const PreparedCudaFastDvdnetTensorContract&,const PreparedFastDvdnetModelContract&)noexcept->std::unique_ptr<IPreparedFastDvdnetOrtSession> override{if(o.provider_policy!=onnx::OrtProviderPolicy::strict_cuda_only||!o.user_compute_stream||o.enable_tensorrt)return{};auto p=std::make_unique<FakeOrt>(ev,fail_run_at);made=p.get();return p;}};
} // namespace

int main() {
  bool ok = true;
  ModelPack pack; pack.task="video_denoise"; pack.id="fastdvdnet_sigma15";
  {
    auto selected=Config(&pack).tensors;
    PreparedFastDvdnetModelContract model;
    model.input_names={"noisy","noise_map"};model.output_name="denoised";
    model.input_channels={15,1};model.output_channels=3;
    model.input_dynamic_hw={true,true};model.output_dynamic_hw=true;
    onnx::OrtSessionInfo info;
    info.input_names={"noisy","noise_map"};info.output_names={"denoised"};
    info.input_shapes={{1,15,-1,-1},{1,1,-1,-1}};
    info.output_shapes={{1,3,-1,-1}};
    info.input_elem_types={1,1};info.output_elem_types={1};
    ok=Check(ValidatePreparedFastDvdnetSessionMetadata(info,model,selected),
             "dynamic manifest metadata must accept setup-selected H/W bindings")&&ok;
    auto bad=info;bad.input_names[0]="input";
    ok=Check(!ValidatePreparedFastDvdnetSessionMetadata(bad,model,selected),"wrong graph input name must fail")&&ok;
    bad=info;bad.input_shapes[0][1]=3;
    ok=Check(!ValidatePreparedFastDvdnetSessionMetadata(bad,model,selected),"wrong graph channel count must fail")&&ok;
    bad=info;bad.input_elem_types[1]=11;
    ok=Check(!ValidatePreparedFastDvdnetSessionMetadata(bad,model,selected),"wrong graph dtype must fail")&&ok;
    model.input_dynamic_hw={false,false};model.output_dynamic_hw=false;
    info.input_shapes={{1,15,8,8},{1,1,8,8}};info.output_shapes={{1,3,8,8}};
    ok=Check(ValidatePreparedFastDvdnetSessionMetadata(info,model,selected),"matching fixed graph geometry must pass")&&ok;
    info.output_shapes[0][3]=12;
    ok=Check(!ValidatePreparedFastDvdnetSessionMetadata(info,model,selected),"wrong fixed graph geometry must fail")&&ok;
  }
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
  config=Config(&pack); rejected_runtime.prepare.ort_evidence = Evidence(config);
  rejected_runtime.prepare.ort_evidence.provider.cpu_ep_fallback_disabled=false;
  rejected_runtime.preserve_prepare_evidence=true;
  ok = Check(rejected.Prepare(config).status==PreparedCudaFastDvdnetStatus::wrong_provider &&
             rejected_runtime.prepares==1 && rejected_runtime.shutdowns==1,
             "CPU fallback evidence from the created session must be rejected and rolled back") && ok;

  NativeCudaState native{}; native_state=&native;
  cuda::CudaDriverApi driver; std::string error;
  ok = Check(driver.InitializeForTesting(NativeFunctions(),&error) && driver.EnsureContextForDevice(0,&error),"native fake CUDA should initialize") && ok;
  ModelFile mf; mf.kind="onnx";mf.role="main";mf.path="fake-valid.onnx";pack.files={mf};
  config=Config(&pack); config.driver=&driver;config.owner_context=native.current;config.owner_stream=reinterpret_cast<cuda::CUstream>(9);
  config.execution_domain.logical_context_generation=driver.context_identity()->logical_generation;
  config.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(config.owner_stream);
  config.available_budget={1ull<<20,1ull<<20,0,0,20,1};
  FakeOrtFactory factory;factory.ev=Evidence(config);
  NativePreparedCudaFastDvdnetRuntime native_runtime(&factory);
  auto np=native_runtime.Prepare(config);
  ok=Check(np.status==PreparedCudaFastDvdnetStatus::ready && native.allocs==6 && factory.made && factory.made->binds==1 && factory.made->runs==1,"native prepare should allocate six tensors, bind once, and warm up")&&ok;
  auto nr=native_runtime.Execute(np.runtime_handle,{1,true,{2,2,2,2,2}},input,&output,15.0f/255.0f);
  ok=Check(nr.status==PreparedCudaFastDvdnetStatus::executed && native.launches==2 && native.copies==2 && factory.made->binds==1 && factory.made->runs==2,"native execute should launch both kernels, fill history, and reuse binding")&&ok;
  auto ns=native_runtime.Shutdown(np.runtime_handle);
  ok=Check(ns.quiescent && native.frees==6,"native shutdown should quiesce and free all six tensors")&&ok;
  np=native_runtime.Prepare(config);
  nr=native_runtime.Execute(np.runtime_handle,{2,true,{2,2,2,2,2}},input,&output,0.1f);
  ok=Check(np.status==PreparedCudaFastDvdnetStatus::ready&&
           nr.status==PreparedCudaFastDvdnetStatus::executed,
           "successful shutdown must reset candidate flags for prepare and execute reuse")&&ok;
  ok=Check(native_runtime.Shutdown(np.runtime_handle).quiescent,
           "reused native runtime should shut down")&&ok;

  {
    NativeCudaState state{};native_state=&state;cuda::CudaDriverApi d;std::string e;
    d.InitializeForTesting(NativeFunctions(),&e);d.EnsureContextForDevice(0,&e);
    auto c=Config(&pack);c.driver=&d;c.owner_context=state.current;c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);
    c.available_budget={1,1,0,0,20,1};FakeOrtFactory f;f.ev=Evidence(c);
    NativePreparedCudaFastDvdnetRuntime r(&f);
    ok=Check(r.Prepare(c).status==PreparedCudaFastDvdnetStatus::resource_exhausted&&state.allocs==0,
             "insufficient declared budget must fail before device allocation")&&ok;
  }

  for(int failed_release=1;failed_release<=3;++failed_release) {
    NativeCudaState state{};native_state=&state;cuda::CudaDriverApi d;std::string e;
    d.InitializeForTesting(NativeFunctions(),&e);d.EnsureContextForDevice(0,&e);
    auto c=Config(&pack);c.driver=&d;c.owner_context=state.current;c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);c.available_budget={1ull<<20,1ull<<20,0,0,20,1};
    FakeOrtFactory f; // Invalid returned evidence forces setup rollback.
    NativePreparedCudaFastDvdnetRuntime r(&f);
    if(failed_release==1)state.fail_free_at=1;
    if(failed_release==2)state.fail_unload_at=1;
    if(failed_release==3)state.fail_event_destroy_at=1;
    auto p=r.Prepare(c);
    ok=Check(p.status==PreparedCudaFastDvdnetStatus::completion_pending&&p.runtime_handle!=0,
             "failed candidate release must expose a cleanup-only handle")&&ok;
    ok=Check(r.Prepare(c).status==PreparedCudaFastDvdnetStatus::malformed_contract,
             "new prepare must not overwrite a retained failed candidate")&&ok;
    const auto s=r.Shutdown(p.runtime_handle);
    ok=Check(s.quiescent&&state.successful_frees==state.allocs,
             "free, module, and event cleanup failures must each succeed on retry")&&ok;
  }

  auto native_failure_case = [&](int stage) {
    NativeCudaState state{}; native_state=&state;
    cuda::CudaDriverApi d; std::string e;
    bool pass=d.InitializeForTesting(NativeFunctions(),&e)&&d.EnsureContextForDevice(0,&e);
    auto c=Config(&pack); c.driver=&d;c.owner_context=state.current;
    c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);
    c.available_budget={1ull<<20,1ull<<20,0,0,20,1};
    FakeOrtFactory f;f.ev=Evidence(c);
    NativePreparedCudaFastDvdnetRuntime r(&f);
    auto p=r.Prepare(c); pass=pass&&p.status==PreparedCudaFastDvdnetStatus::ready;
    const auto allocations=r.Telemetry().allocation_requests;
    if(stage==1)state.fail_launch_at=state.launches+1;
    if(stage==2)state.fail_copy_at=state.copies+1;
    if(stage==3)f.made->fail_run_at=f.made->runs+1;
    if(stage==4)state.fail_launch_at=state.launches+2;
    if(stage==5)state.fail_record_at=state.records+1;
    const auto fr=r.Execute(p.runtime_handle,{1,true,{2,2,2,2,2}},input,&output,0.1f);
    pass=pass&&fr.status!=PreparedCudaFastDvdnetStatus::executed&&state.syncs==1;
    const auto s=r.Shutdown(p.runtime_handle);
    pass=pass&&s.quiescent&&state.syncs==2&&state.frees==state.allocs&&
         r.Telemetry().allocation_requests==allocations;
    return pass;
  };
  for(int stage=1;stage<=5;++stage)
    ok=Check(native_failure_case(stage),"each partial frame failure must retain then quiesce native work without hot-path allocation")&&ok;

  {
    NativeCudaState state{}; native_state=&state;
    cuda::CudaDriverApi d; std::string e;
    ok=Check(d.InitializeForTesting(NativeFunctions(),&e)&&d.EnsureContextForDevice(0,&e),"warmup failure driver setup")&&ok;
    auto c=Config(&pack);c.driver=&d;c.owner_context=state.current;c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);
    c.available_budget={1ull<<20,1ull<<20,0,0,20,1};
    FakeOrtFactory f;f.ev=Evidence(c);f.fail_run_at=1;
    NativePreparedCudaFastDvdnetRuntime r(&f);
    state.fail_syncs=2;
    auto p=r.Prepare(c);
    ok=Check(p.status==PreparedCudaFastDvdnetStatus::completion_pending&&p.runtime_handle!=0&&state.frees==0,
             "failed warmup synchronization must retain a retryable owner")&&ok;
    ok=Check(r.Prepare(c).status==PreparedCudaFastDvdnetStatus::malformed_contract,
             "prepare must reject while warmup work remains pending")&&ok;
    auto s=r.Shutdown(p.runtime_handle);
    ok=Check(!s.quiescent&&state.frees==0,
             "failed warmup cleanup must remain retryable")&&ok;
    s=r.Shutdown(p.runtime_handle);
    ok=Check(s.quiescent&&state.frees==state.allocs,
             "explicit shutdown retry must quiesce and release failed warmup")&&ok;
  }

  for(int failed_memset=1;failed_memset<=2;++failed_memset) {
    NativeCudaState state{};native_state=&state;cuda::CudaDriverApi d;std::string e;
    d.InitializeForTesting(NativeFunctions(),&e);d.EnsureContextForDevice(0,&e);
    auto c=Config(&pack);c.driver=&d;c.owner_context=state.current;c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);c.available_budget={1ull<<20,1ull<<20,0,0,20,1};
    FakeOrtFactory f;f.ev=Evidence(c);NativePreparedCudaFastDvdnetRuntime r(&f);
    state.fail_memset_at=failed_memset;auto p=r.Prepare(c);
    ok=Check(p.status==PreparedCudaFastDvdnetStatus::inference_failed&&state.syncs==1&&state.frees==state.allocs,
             "each partial warmup memset failure must synchronize before release")&&ok;
  }

  {
    NativeCudaState state{}; native_state=&state;
    cuda::CudaDriverApi d;std::string e;d.InitializeForTesting(NativeFunctions(),&e);d.EnsureContextForDevice(0,&e);
    auto c=Config(&pack);c.driver=&d;c.owner_context=state.current;c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);c.available_budget={1ull<<20,1ull<<20,0,0,20,1};
    std::string noisy="noisy",noise="noise_map",den="denoised";
    c.tensors.noisy_name=noisy.c_str();c.tensors.noise_map_name=noise.c_str();c.tensors.denoised_name=den.c_str();
    FakeOrtFactory f;f.ev=Evidence(c);NativePreparedCudaFastDvdnetRuntime r(&f);auto p=r.Prepare(c);
    noisy.assign("destroyed");noise.assign("destroyed");den.assign("destroyed");
    auto fr=r.Execute(p.runtime_handle,{1,true,{2,2,2,2,2}},input,&output,0.1f);
    ok=Check(fr.status==PreparedCudaFastDvdnetStatus::executed&&f.made->binding_valid,
             "native bindings must own tensor names and shapes beyond caller config lifetime")&&ok;
    state.not_ready_queries=1;
    auto s=r.Shutdown(p.runtime_handle);
    ok=Check(!s.quiescent&&state.frees==0,"event-pending shutdown must retain all owners")&&ok;
    s=r.Shutdown(p.runtime_handle);
    ok=Check(s.quiescent&&state.frees==state.allocs,"event cleanup must be retryable")&&ok;
  }
  {
    NativeCudaState state{};native_state=&state;cuda::CudaDriverApi d;std::string e;
    d.InitializeForTesting(NativeFunctions(),&e);d.EnsureContextForDevice(0,&e);
    auto c=Config(&pack);c.driver=&d;c.owner_context=state.current;c.owner_stream=reinterpret_cast<cuda::CUstream>(9);
    c.execution_domain.logical_context_generation=d.context_identity()->logical_generation;
    c.execution_domain.queue_or_stream_identity=reinterpret_cast<std::uintptr_t>(c.owner_stream);c.available_budget={1ull<<20,1ull<<20,0,0,20,1};
    FakeOrtFactory f;f.ev=Evidence(c);NativePreparedCudaFastDvdnetRuntime r(&f);auto p=r.Prepare(c);
    cuda::CudaDriverApi replacement;replacement.InitializeForTesting(NativeFunctions(),&e);replacement.EnsureContextForDevice(0,&e);d=std::move(replacement);
    auto fr=r.Execute(p.runtime_handle,{1,true,{2,2,2,2,2}},input,&output,0.1f);
    ok=Check(fr.status==PreparedCudaFastDvdnetStatus::device_lost&&state.launches==0,
             "execute must bind the exact prepared context identity")&&ok;
    ok=Check(r.Shutdown(p.runtime_handle).quiescent,"replaced-context test cleanup")&&ok;
  }
  return ok ? 0 : 1;
}
