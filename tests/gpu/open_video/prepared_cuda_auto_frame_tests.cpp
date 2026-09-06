#include "core/open_video/prepared_cuda_auto_frame.h"

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using namespace studiocast;
using namespace studiocast::open_video;
bool Check(bool v, std::string_view m) {
  if (!v)
    std::cerr << "[FAIL] " << m << '\n';
  return v;
}

struct FakeCrop final : IPreparedCudaAutoFrameCropRuntime {
  bool prepare_ok = true, crop_ok = true, shutdown_ok = true;
  int crops = 0, shutdowns = 0;
  PreparedCudaAutoFrameRect last{};
  cuda::CudaImage output{};
  bool Prepare(const PreparedCudaAutoFrameBinding &, int w, int h, std::size_t,
               std::string *) noexcept override {
    output.ptr = 0x9000;
    output.pitch = w * 3;
    output.w = w;
    output.h = h;
    output.format = cuda::PixelFormatGpu::rgb_u8;
    return prepare_ok;
  }
  bool Crop(const cuda::CudaImage &, const PreparedCudaAutoFrameRect &r,
            const cuda::CudaImage **o, std::string *) noexcept override {
    ++crops;
    last = r;
    if (!crop_ok)
      return false;
    if (o)
      *o = &output;
    return true;
  }
  bool Shutdown(std::string *) noexcept override {
    ++shutdowns;
    return shutdown_ok;
  }
};
struct Analysis final : IPreparedCudaAutoFrameAnalysis {
  std::vector<PreparedCudaAutoFrameAnalysisResult> results;
  std::size_t at = 0;
  int calls = 0;
  PreparedCudaAutoFrameAnalysisResult
  Analyze(const PreparedCudaAutoFrameBinding &, const cuda::CudaImage &,
          std::uint64_t, FrameAnalysisCache *,
          const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept
      override {
    ++calls;
    return results[std::min(at++, results.size() - 1)];
  }
};
PreparedCudaAutoFrameBinding Binding() {
  static cuda::CudaDriverApi driver;
  PreparedCudaAutoFrameBinding b;
  b.driver = &driver;
  b.owner_context = reinterpret_cast<cuda::CUcontext>(0x10);
  b.stream = reinterpret_cast<cuda::CUstream>(0x20);
  b.stream_identity = 3;
  b.owner_identity = 4;
  b.configuration_generation = 5;
  b.model_generation = 6;
  b.context.physical_device.uuid[0] = 1;
  b.context.physical_device.pci_bus_id[0] = 'x';
  b.context.physical_device.pci_bus_id_length = 1;
  b.context.retained_context_id = 7;
  b.context.logical_generation = 8;
  return b;
}
cuda::CudaImage Image() {
  cuda::CudaImage i;
  i.ptr = 0x1000;
  i.pitch = 1920;
  i.w = 640;
  i.h = 480;
  i.format = cuda::PixelFormatGpu::rgb_u8;
  return i;
}
PreparedCudaAutoFrameSetup Setup(Analysis *f, Analysis *m,
                                 FrameAnalysisCache *c) {
  PreparedCudaAutoFrameSetup s;
  s.binding = Binding();
  s.frame_width = 640;
  s.frame_height = 480;
  s.face = f;
  s.matte = m;
  s.analysis_cache = c;
  s.resource_budget_bytes = 640 * 480 * 3;
  return s;
}
PreparedCudaAutoFrameAnalysisResult Box(PreparedCudaAutoFrameSource source,
                                        float x, float y, float w, float h) {
  PreparedCudaAutoFrameAnalysisResult r;
  r.ok = r.found = true;
  r.box = {source, {x, y, w, h}, .9F};
  return r;
}

bool GeometryAndSequenceReset() {
  FakeCrop crop;
  Analysis face;
  FrameAnalysisCache cache;
  face.results = {Box(PreparedCudaAutoFrameSource::face, 100, 80, 100, 120),
                  Box(PreparedCudaAutoFrameSource::face, 300, 80, 100, 120),
                  Box(PreparedCudaAutoFrameSource::face, 100, 80, 100, 120)};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  std::string e;
  auto s = Setup(&face, nullptr, &cache);
  s.knobs = {50, 50, .15F};
  bool ok = Check(af.Prepare(s, &h, &e), "prepare");
  auto image = Image();
  auto first = af.Execute(h, s.binding, 10, image);
  auto second = af.Execute(h, s.binding, 11, image);
  auto reset = af.Execute(h, s.binding, 20, image);
  ok &= Check(first.status == PreparedCudaAutoFrameStatus::executed &&
                  crop.crops == 3,
              "face frames did not crop");
  ok &= Check(second.source_crop.x > first.source_crop.x &&
                  second.source_crop.x < reset.source_crop.x + 100,
              "smoothing not applied");
  ok &= Check(std::abs(reset.source_crop.x - first.source_crop.x) < .01F,
              "discontinuity did not reset smoothing");
  ok &= Check(first.source_crop.w / first.source_crop.h > 1.332F &&
                  first.source_crop.w / first.source_crop.h < 1.334F,
              "capture crop aspect drifted");
  auto stale = s.binding;
  ++stale.model_generation;
  ok &= Check(af.Execute(h, stale, 21, image).status ==
                  PreparedCudaAutoFrameStatus::stale_binding,
              "stale model binding accepted");
  return ok;
}

bool MatteCadenceCountersAndFailures() {
  FakeCrop crop;
  Analysis matte;
  FrameAnalysisCache cache;
  auto found = Box(PreparedCudaAutoFrameSource::matte, 120, 100, 160, 180);
  found.transfers.intermediate_readbacks = 1;
  found.transfers.synchronizations = 1;
  found.transfers.cpu_tail_boundaries = 1;
  found.completion_waits = 1;
  found.cpu_tail_boundaries = 1;
  matte.results = {found, found};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  auto s = Setup(nullptr, &matte, &cache);
  std::string e;
  bool ok = af.Prepare(s, &h, &e);
  auto image = Image();
  auto a = af.Execute(h, s.binding, 1, image);
  auto b = af.Execute(h, s.binding, 2, image);
  auto c = af.Execute(h, s.binding, 11, image);
  ok &= Check(matte.calls == 2 && a.analysis_ran && b.analysis_ran == false &&
                  c.analysis_ran,
              "matte cadence/cache mismatch");
  ok &= Check(a.transfers.intermediate_readbacks == 1 &&
                  a.completion_waits == 1 && a.cpu_tail_boundaries == 1,
              "actual analysis counters lost");
  crop.crop_ok = false;
  auto drop = af.Execute(h, s.binding, 12, image);
  ok &= Check(drop.status == PreparedCudaAutoFrameStatus::drop_frame &&
                  !drop.output,
              "partial crop failure did not drop");
  return ok;
}

bool StrictFailureAndRetryableShutdown() {
  FakeCrop crop;
  Analysis face, matte;
  FrameAnalysisCache cache;
  PreparedCudaAutoFrameAnalysisResult failed;
  failed.pending_work = false;
  failed.transfers.intermediate_readbacks = 4;
  face.results = {failed};
  matte.results = {Box(PreparedCudaAutoFrameSource::matte, 1, 1, 20, 20)};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  auto s = Setup(&face, &matte, &cache);
  std::string e;
  bool ok = af.Prepare(s, &h, &e);
  auto image = Image();
  auto r = af.Execute(h, s.binding, 1, image);
  ok &= Check(r.status == PreparedCudaAutoFrameStatus::preserved_input &&
                  r.valid_input_preserved && matte.calls == 0 &&
                  r.transfers.intermediate_readbacks == 4,
              "strict face failure fell back");
  crop.shutdown_ok = false;
  ok &= Check(!af.Shutdown(&e), "shutdown failure hidden");
  crop.shutdown_ok = true;
  ok &= Check(af.Shutdown(&e) && crop.shutdowns == 2, "shutdown not retryable");
  return ok;
}

bool MatteBoxReference() {
  std::vector<float> alpha(20 * 20);
  for (int y = 4; y < 16; ++y)
    for (int x = 5; x < 15; ++x)
      alpha[y * 20 + x] = 1;
  PreparedCudaAutoFrameTrackingBox b;
  bool ok = Check(PreparedCudaAutoFrameMatteAdapter::ComputeBox(
                      alpha.data(), 20, 20, 200, 100, &b) &&
                      b.box.x == 50 && b.box.y == 20 && b.box.w == 100 &&
                      b.box.h == 60,
                  "matte bbox changed");
  FrameMatteArtifactKey key;
  key.provider_id = "open_cuda";
  key.model_id = "matte-v1";
  key.storage = FrameMatteStorage::cpu_f32_alpha;
  key.frame_width = 200;
  key.frame_height = 100;
  key.matte_width = 20;
  key.matte_height = 20;
  key.device_context = 7;
  key.stream = 3;
  FrameArtifactCache artifacts;
  FrameMatteArtifact artifact;
  artifact.key = key;
  artifact.cpu_alpha = alpha;
  (void)artifacts.StoreMatte(42, std::move(artifact));
  FrameAnalysisCache cache;
  cache.capture_sequence = 42;
  cache.artifacts = std::move(artifacts);
  PreparedCudaAutoFrameMatteAdapter adapter(nullptr, nullptr, key, 200, 100);
  auto binding = Binding();
  auto image = Image();
  const auto shared = adapter.Analyze(binding, image, 42, &cache, {});
  ok &= Check(adapter.HasSameFrameArtifact(42, &cache) && shared.ok &&
                  shared.found && shared.box.box.x == 50 &&
                  shared.transfers.intermediate_readbacks == 0,
              "compatible same-frame matte artifact was not reused");
  key.model_id = "other";
  PreparedCudaAutoFrameMatteAdapter stale(nullptr, nullptr, key, 200, 100);
  ok &= Check(!stale.HasSameFrameArtifact(42, &cache) &&
                  !stale.Analyze(binding, image, 42, &cache, {}).ok,
              "foreign matte artifact was accepted");
  return ok;
}
} // namespace
int main() {
  return GeometryAndSequenceReset() && MatteCadenceCountersAndFailures() &&
                 StrictFailureAndRetryableShutdown() && MatteBoxReference()
             ? 0
             : 1;
}
