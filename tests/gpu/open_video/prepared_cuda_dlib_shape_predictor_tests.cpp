#include "core/cuda/kernels/dlib_shape_predictor.h"
#include "core/open_video/prepared_cuda_dlib_shape_predictor.h"
#include <cstdlib>
#include <cstring>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/serialize.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>
using namespace studiocast::cuda;
namespace {
CUcontext ctx = reinterpret_cast<CUcontext>(1);
int loads, gets, launches, unloads;
CUresult launch_result = CUDA_SUCCESS;
CUresult copy_result = CUDA_SUCCESS;
int allocations, frees, syncs, fail_sync_from, fail_sync_through;
int fail_free_from, fail_free_through;
int fail_alloc_from, fail_alloc_through;
std::size_t tensor_pitch_extra;
const volatile std::byte *pending_copy_source;
bool inspect_pending_copy_source;
CUresult Current(CUcontext *c) {
  *c = ctx;
  return 0;
}
CUresult Load(CUmodule *m, const void *) {
  ++loads;
  *m = reinterpret_cast<CUmodule>(2);
  return 0;
}
CUresult Get(CUfunction *f, CUmodule, const char *) {
  ++gets;
  *f = reinterpret_cast<CUfunction>(3);
  return 0;
}
CUresult Unload(CUmodule) {
  ++unloads;
  return 0;
}
CUresult Launch(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                unsigned, unsigned, CUstream, void **, void **) {
  ++launches;
  return launch_result;
}
CUresult Alloc(CUdeviceptr *p, std::size_t *q, std::size_t n, std::size_t,
               unsigned) {
  ++allocations;
  if (allocations >= fail_alloc_from && allocations <= fail_alloc_through) {
    *p = 0;
    *q = 0;
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *p = static_cast<CUdeviceptr>(0x1000 + allocations * 0x1000);
  *q = n == 8192 ? n : n + tensor_pitch_extra;
  return 0;
}
CUresult Free(CUdeviceptr) {
  ++frees;
  return frees >= fail_free_from && frees <= fail_free_through
             ? CUDA_ERROR_LAUNCH_FAILED
             : CUDA_SUCCESS;
}
CUresult Copy(const CUDA_MEMCPY2D *copy, CUstream) {
  if (copy_result != CUDA_SUCCESS)
    pending_copy_source = static_cast<const std::byte *>(copy->srcHost);
  return copy_result;
}
CUresult Set(CUdeviceptr, unsigned char, std::size_t, CUstream) { return 0; }
CUresult SC(CUstream *s, unsigned) {
  *s = reinterpret_cast<CUstream>(4);
  return 0;
}
CUresult S(CUstream) {
  if (inspect_pending_copy_source)
    (void)*pending_copy_source;
  ++syncs;
  return syncs >= fail_sync_from && syncs <= fail_sync_through
             ? CUDA_ERROR_LAUNCH_FAILED
             : CUDA_SUCCESS;
}
CUresult DC(int *n) {
  *n = 1;
  return 0;
}
CUresult DG(CUdevice *d, int n) {
  *d = n;
  return 0;
}
CUresult DU(CUuuid *u, CUdevice) {
  std::memset(u, 0, sizeof(*u));
  u->bytes[0] = 1;
  return 0;
}
CUresult DR(CUcontext *c, CUdevice) {
  *c = ctx;
  return 0;
}
CUresult DD(CUdevice) { return 0; }
CUresult CS(CUcontext c) {
  ctx = c;
  return 0;
}
CUresult CD(CUdevice *d) {
  *d = 0;
  return 0;
}
bool Need(bool x, const char *m) {
  if (!x)
    std::cerr << m << '\n';
  return x;
}
std::filesystem::path TinyPack() {
  auto root = std::filesystem::temp_directory_path() /
              ("studiocast-dlib-" + std::to_string(::getpid()));
  auto dir = root / "face_landmarks" / "tiny-dlib68";
  std::filesystem::create_directories(dir);
  dlib::matrix<float, 0, 1> shape(136);
  for (long i = 0; i < 68; ++i) {
    shape(2 * i) = 0.2f + 0.005f * i;
    shape(2 * i + 1) = 0.3f + 0.004f * i;
  }
  dlib::impl::regression_tree tree;
  tree.splits.push_back({0, 1, 0.0f});
  tree.leaf_values.emplace_back(136);
  tree.leaf_values.emplace_back(136);
  tree.leaf_values[0] = 0;
  tree.leaf_values[1] = 0;
  std::vector<std::vector<dlib::impl::regression_tree>> forests{{tree}};
  std::vector<std::vector<dlib::vector<float, 2>>> pixels{
      {{0.5f, 0.5f}, {0.25f, 0.25f}}};
  dlib::shape_predictor predictor(shape, forests, pixels);
  dlib::serialize((dir / "model.dat").string()) << predictor;
  std::ofstream(dir / "LICENSE.txt") << "synthetic test fixture\n";
  std::ofstream(dir / "model.json")
      << R"({"schema_version":2,"id":"tiny-dlib68","display_name":"tiny","task":"face_landmarks","files":[{"name":"model.dat","kind":"dlib_shape_predictor","role":"main","sha256":""}],"dlib":{"predictor_kind":"shape_predictor_68_face_landmarks","num_landmarks":68}})";
  return root;
}
std::filesystem::path OversizedPack() {
  auto root = std::filesystem::temp_directory_path() /
              ("studiocast-dlib-oversized-" + std::to_string(::getpid()));
  auto dir = root / "face_landmarks" / "oversized-dlib68";
  std::filesystem::create_directories(dir);
  std::ofstream artifact(dir / "model.dat", std::ios::binary);
  int version = 1;
  long nr = -136, nc = -1;
  dlib::serialize(version, artifact);
  dlib::serialize(nr, artifact);
  dlib::serialize(nc, artifact);
  for (int i = 0; i < 136; ++i)
    dlib::serialize(0.0f, artifact);
  dlib::serialize(std::numeric_limits<unsigned long>::max(), artifact);
  artifact.close();
  std::ofstream(dir / "LICENSE.txt") << "synthetic test fixture\n";
  std::ofstream(dir / "model.json")
      << R"({"schema_version":2,"id":"oversized-dlib68","display_name":"oversized","task":"face_landmarks","files":[{"name":"model.dat","kind":"dlib_shape_predictor","role":"main","sha256":""}],"dlib":{"predictor_kind":"shape_predictor_68_face_landmarks","num_landmarks":68}})";
  return root;
}
} // namespace
int main() {
  CudaDriverApi d;
  CudaDriverApi::Functions f{};
  f.cuInit = [](unsigned) { return 0; };
  f.cuModuleLoadData = Load;
  f.cuModuleGetFunction = Get;
  f.cuModuleUnload = Unload;
  f.cuLaunchKernel = Launch;
  f.cuMemAllocPitch = Alloc;
  f.cuMemFree = Free;
  f.cuMemcpy2DAsync = Copy;
  f.cuMemsetD8Async = Set;
  f.cuStreamCreate = SC;
  f.cuStreamDestroy = S;
  f.cuStreamSynchronize = S;
  f.cuDeviceGetCount = DC;
  f.cuDeviceGet = DG;
  f.cuDeviceGetUuid = DU;
  f.cuDevicePrimaryCtxRetain = DR;
  f.cuDevicePrimaryCtxRelease = DD;
  f.cuCtxSetCurrent = CS;
  f.cuCtxGetCurrent = Current;
  f.cuCtxGetDevice = CD;
  std::string e;
  bool ok = Need(d.InitializeForTesting(f, &e), e.c_str());
  kernels::PreparedDlibShapePredictorKernel k;
  auto s = reinterpret_cast<CUstream>(4);
  ok &= Need(kernels::PrepareDlibShapePredictorKernel(d, ctx, s, &k, &e),
             e.c_str());
  CudaImage image;
  image.ptr = 100;
  image.pitch = 30;
  image.w = 10;
  image.h = 10;
  image.format = PixelFormatGpu::rgb_u8;
  ok &= Need(kernels::LaunchDlibShapePredictor(k, image, 7, 0, 0, 7, 1, 1, 1, 2,
                                               3, 4, 5, 6, 7, 8, 9, 1, 10, 11,
                                               12, &e),
             e.c_str());
  launch_result = CUDA_ERROR_LAUNCH_FAILED;
  ok &= Need(!kernels::LaunchDlibShapePredictor(k, image, 7, 0, 0, 7, 1, 1, 1,
                                                2, 3, 4, 5, 6, 7, 8, 9, 1, 10,
                                                11, 12, &e),
             "launch failure must be reported");
  ctx = reinterpret_cast<CUcontext>(9);
  ok &= Need(!kernels::DestroyDlibShapePredictorKernel(&k, &e) && k.module,
             "failed shutdown must retain module");
  ctx = reinterpret_cast<CUcontext>(1);
  ok &= Need(kernels::DestroyDlibShapePredictorKernel(&k, &e) && !k.module,
             "retry shutdown must release module");
  ok &= Need(loads == 1 && gets == 1 && launches == 2 && unloads == 1,
             "unexpected lifecycle counts");
  const auto root = TinyPack();
  CudaPhysicalDeviceIdentity physical;
  physical.uuid[0] = 1;
  CudaContextIdentity identity{physical, 1, 1};
  studiocast::open_video::PreparedCudaDlibBinding binding{&d, identity, ctx, s,
                                                          1,  2,        3};
  studiocast::open_video::PreparedCudaDlibSetup setup{binding, root,
                                                      "tiny-dlib68", 1u << 20};
  studiocast::open_video::PreparedCudaDlibShapePredictor predictor;
  const int allocations_before = allocations;
  syncs = 0;
  fail_sync_from = 2;
  fail_sync_through = 3;
  ok &= Need(!predictor.Prepare(setup, &e),
             "injected setup synchronization must fail");
  if (allocations == allocations_before)
    std::cerr << "Prepare failed before CUDA injection: " << e << '\n';
  ok &= Need(allocations > allocations_before,
             "production Prepare did not reach injected CUDA upload");
  const int retained_allocations = allocations;
  const int retained_frees = frees;
  ok &= Need(retained_frees == 0, "pending setup failure released resources");
  ok &=
      Need(!predictor.Prepare(setup, &e) && allocations == retained_allocations,
           "quarantined candidate allowed another Prepare allocation");
  const bool first_shutdown = predictor.Shutdown(&e);
  if (first_shutdown || frees != retained_frees)
    std::cerr << "shutdown evidence success=" << first_shutdown
              << " frees=" << frees << " retained=" << retained_frees
              << " syncs=" << syncs << '\n';
  ok &= Need(!first_shutdown && frees == retained_frees,
             "failed shutdown released pending resources");
  fail_sync_from = 0;
  fail_sync_through = 0;
  ok &= Need(predictor.Shutdown(&e) &&
                 frees == retained_allocations - allocations_before,
             "shutdown retry did not release complete candidate");
  std::filesystem::remove_all(root);
  const auto padded_root = TinyPack();
  studiocast::open_video::PreparedCudaDlibShapePredictor padded;
  setup.models_root = padded_root;
  setup.model_id = "tiny-dlib68";
  tensor_pitch_extra = 1u << 20;
  const int before_padded_allocations = allocations;
  const int before_padded_frees = frees;
  ok &= Need(!padded.Prepare(setup, &e),
             "pitched allocations exceeding budget must fail");
  ok &= Need(frees - before_padded_frees ==
                 allocations - before_padded_allocations,
             "budget rejection leaked padded allocations");
  tensor_pitch_extra = 0;
  std::filesystem::remove_all(padded_root);
  const auto cleanup_root = TinyPack();
  studiocast::open_video::PreparedCudaDlibShapePredictor failed_cleanup;
  setup.models_root = cleanup_root;
  setup.model_id = "tiny-dlib68";
  tensor_pitch_extra = 1u << 20;
  fail_free_from = frees + 1;
  fail_free_through = fail_free_from;
  const int before_cleanup_allocations = allocations;
  ok &= Need(!failed_cleanup.Prepare(setup, &e),
             "budget failure with failed cleanup must reject setup");
  const int after_failed_cleanup_allocations = allocations;
  const int after_failed_cleanup_frees = frees;
  ok &= Need(!failed_cleanup.Prepare(setup, &e) &&
                 after_failed_cleanup_allocations > before_cleanup_allocations &&
                 allocations == after_failed_cleanup_allocations,
             "failed setup cleanup did not retain retry state");
  fail_free_from = 0;
  fail_free_through = 0;
  ok &= Need(failed_cleanup.Shutdown(&e) &&
                 frees > after_failed_cleanup_frees,
             "failed setup cleanup was not retryable");
  tensor_pitch_extra = 0;
  std::filesystem::remove_all(cleanup_root);
  const auto later_alloc_root = TinyPack();
  studiocast::open_video::PreparedCudaDlibShapePredictor later_alloc;
  setup.models_root = later_alloc_root;
  fail_alloc_from = allocations + 2;
  fail_alloc_through = fail_alloc_from;
  fail_free_from = frees + 1;
  fail_free_through = fail_free_from;
  const int before_later_alloc = allocations;
  ok &= Need(!later_alloc.Prepare(setup, &e) &&
                 allocations == before_later_alloc + 2,
             "later table allocation failure did not reject setup");
  const int after_later_alloc = allocations;
  const int after_later_free_failure = frees;
  ok &= Need(!later_alloc.Prepare(setup, &e) &&
                 allocations == after_later_alloc,
             "later allocation failure lost cleanup retry state");
  fail_alloc_from = 0;
  fail_alloc_through = 0;
  fail_free_from = 0;
  fail_free_through = 0;
  ok &= Need(later_alloc.Shutdown(&e) && frees > after_later_free_failure,
             "earlier table free was not retried after allocation failure");
  std::filesystem::remove_all(later_alloc_root);
  const auto partial_root = TinyPack();
  studiocast::open_video::PreparedCudaDlibShapePredictor partial;
  setup.models_root = partial_root;
  setup.model_id = "tiny-dlib68";
  copy_result = CUDA_ERROR_LAUNCH_FAILED;
  const int before_partial_allocations = allocations;
  const int before_partial_frees = frees;
  ok &= Need(!partial.Prepare(setup, &e) &&
                 allocations == before_partial_allocations + 1 &&
                 frees == before_partial_frees,
             "partial upload failure released pending storage");
  ok &= Need(!partial.Prepare(setup, &e) &&
                 allocations == before_partial_allocations + 1,
             "partial upload failure did not quarantine candidate");
  fail_sync_from = syncs + 1;
  fail_sync_through = fail_sync_from;
  inspect_pending_copy_source = true;
  ok &= Need(!partial.Shutdown(&e) && frees == before_partial_frees,
             "failed partial-upload shutdown released pending storage");
  fail_sync_from = 0;
  fail_sync_through = 0;
  inspect_pending_copy_source = false;
  copy_result = CUDA_SUCCESS;
  ok &= Need(partial.Shutdown(&e) && frees == before_partial_frees + 1,
             "partial-upload shutdown retry did not release storage");
  std::filesystem::remove_all(partial_root);
  const auto launch_root = TinyPack();
  studiocast::open_video::PreparedCudaDlibShapePredictor failed_launch;
  setup.models_root = launch_root;
  const int before_launch_frees = frees;
  ok &= Need(failed_launch.Prepare(setup, &e), e.c_str());
  launch_result = CUDA_ERROR_LAUNCH_FAILED;
  const auto failed = failed_launch.Execute(binding, image, {1, 1, 8, 8}, 1,
                                             &e);
  ok &= Need(failed.status ==
                 studiocast::open_video::PreparedCudaDlibStatus::launch_failed,
             "failed launch was not reported");
  ok &= Need(failed.pending_work,
             "failed launch did not expose possibly pending work");
  ok &= Need(!failed_launch.ready(),
             "quarantined failed launch still advertised readiness");
  ok &= Need(failed_launch.Execute(binding, image, {1, 1, 8, 8}, 2, &e).status ==
                 studiocast::open_video::PreparedCudaDlibStatus::shutdown,
             "failed launch did not quarantine subsequent execution");
  fail_sync_from = syncs + 1;
  fail_sync_through = fail_sync_from;
  ok &= Need(!failed_launch.Shutdown(&e) && frees == before_launch_frees,
             "failed launch shutdown released possibly pending storage");
  fail_sync_from = 0;
  fail_sync_through = 0;
  ok &= Need(failed_launch.Shutdown(&e) && frees > before_launch_frees,
             "failed launch shutdown retry did not release storage");
  launch_result = CUDA_SUCCESS;
  std::filesystem::remove_all(launch_root);
  const auto oversized_root = OversizedPack();
  studiocast::open_video::PreparedCudaDlibShapePredictor oversized;
  setup.models_root = oversized_root;
  setup.model_id = "oversized-dlib68";
  const int before_oversized = allocations;
  ok &= Need(!oversized.Prepare(setup, &e) && allocations == before_oversized,
             "oversized wire cardinality reached GPU allocation");
  std::filesystem::remove_all(oversized_root);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
