#include "core/open_video/prepared_cuda_dlib_shape_predictor.h"
#include "core/cuda/kernels/dlib_shape_predictor.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#ifndef STUDIOCAST_HAVE_DLIB
#define STUDIOCAST_HAVE_DLIB 0
#endif
#if STUDIOCAST_HAVE_DLIB
#include <dlib/image_processing/shape_predictor.h>
#endif

namespace studiocast::open_video {
namespace {
bool Same(const PreparedCudaDlibBinding &a, const PreparedCudaDlibBinding &b) {
  return a.driver == b.driver && a.context == b.context &&
         a.owner_context == b.owner_context && a.stream == b.stream &&
         a.stream_identity == b.stream_identity &&
         a.owner_identity == b.owner_identity &&
         a.configuration_generation == b.configuration_generation;
}
struct Buffer {
  cuda::CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
};
#if STUDIOCAST_HAVE_DLIB
bool Charge(std::size_t count, std::size_t size, std::size_t budget,
            std::size_t *used) {
  if (*used > budget || count > (budget - *used) / size)
    return false;
  *used += count * size;
  return true;
}
bool ValidatePredictorWire(std::istream &in, std::size_t budget,
                           std::string *error) {
  try {
    int version = 0;
    long nr = 0, nc = 0;
    dlib::deserialize(version, in);
    dlib::deserialize(nr, in);
    dlib::deserialize(nc, in);
    if (version != 1 || nr != -136 || nc != -1)
      throw dlib::serialization_error("version/initial shape");
    float scalar = 0;
    for (int i = 0; i < 136; ++i)
      dlib::deserialize(scalar, in);
    std::size_t used = 136 * sizeof(float);
    unsigned long cascades = 0;
    dlib::deserialize(cascades, in);
    if (cascades == 0 || cascades > 1024)
      throw dlib::serialization_error("cascade count");
    for (unsigned long c = 0; c < cascades; ++c) {
      unsigned long trees = 0;
      dlib::deserialize(trees, in);
      if (trees == 0 || trees > 100000)
        throw dlib::serialization_error("tree count");
      for (unsigned long t = 0; t < trees; ++t) {
        unsigned long splits = 0;
        dlib::deserialize(splits, in);
        if (!Charge(splits, sizeof(unsigned) * 2 + sizeof(float), budget,
                    &used))
          throw dlib::serialization_error("split budget");
        for (unsigned long i = 0; i < splits; ++i) {
          unsigned long a = 0, b = 0;
          dlib::deserialize(a, in);
          dlib::deserialize(b, in);
          dlib::deserialize(scalar, in);
        }
        unsigned long leaves = 0;
        dlib::deserialize(leaves, in);
        if (leaves != splits + 1 || (leaves & (leaves - 1)) ||
            !Charge(leaves, 136 * sizeof(float), budget, &used))
          throw dlib::serialization_error("leaf count/budget");
        for (unsigned long i = 0; i < leaves; ++i) {
          dlib::deserialize(nr, in);
          dlib::deserialize(nc, in);
          if (nr != -136 || nc != -1)
            throw dlib::serialization_error("leaf shape");
          for (int j = 0; j < 136; ++j)
            dlib::deserialize(scalar, in);
        }
      }
    }
    for (int section = 0; section < 2; ++section) {
      unsigned long outer = 0;
      dlib::deserialize(outer, in);
      if (outer != cascades)
        throw dlib::serialization_error("cascade metadata count");
      for (unsigned long c = 0; c < outer; ++c) {
        unsigned long count = 0;
        dlib::deserialize(count, in);
        if (count == 0 ||
            !Charge(count, section == 0 ? sizeof(unsigned) : 2 * sizeof(float),
                    budget, &used))
          throw dlib::serialization_error("feature metadata budget");
        for (unsigned long i = 0; i < count; ++i) {
          if (section == 0) {
            unsigned long anchor = 0;
            dlib::deserialize(anchor, in);
          } else {
            dlib::deserialize(scalar, in);
            dlib::deserialize(scalar, in);
          }
        }
      }
    }
    if (in.peek() != std::char_traits<char>::eof())
      throw dlib::serialization_error("trailing bytes");
    return true;
  } catch (const std::exception &x) {
    if (error)
      *error =
          std::string("invalid/budget-exceeding dlib predictor: ") + x.what();
    return false;
  }
}
#endif
bool AllocUpload(cuda::CudaDriverApi *d, cuda::CUstream s, const void *p,
                 std::size_t n, Buffer *out, bool *pending, std::string *e) {
  if (!n)
    return false;
  constexpr std::size_t row_bytes = 8192;
  const std::size_t rows = (n + row_bytes - 1) / row_bytes;
  std::vector<std::byte> staging(rows * row_bytes);
  std::memcpy(staging.data(), p, n);
  cuda::CudaDriverApi::PitchAllocation a{};
  if (!d->AllocatePitch(row_bytes, rows, &a, e, 4) || a.pitch != row_bytes) {
    if (a.ptr) {
      std::string ignored;
      (void)d->Free(a.ptr, &ignored);
    }
    if (e && e->empty())
      *e = "dlib table allocation is not contiguous";
    return false;
  }
  out->ptr = a.ptr;
  out->bytes = n;
  if (!d->MemcpyHtoD2DAsync(a.ptr, a.pitch, staging.data(), row_bytes,
                            row_bytes, rows, s, e)) {
    std::string x;
    (void)d->Free(a.ptr, &x);
    out->ptr = 0;
    return false;
  }
  // Setup owns the temporary deserialization vectors.  Complete each upload
  // before its host storage can unwind on a later validation/allocation error.
  if (d->f().cuStreamSynchronize(s) != cuda::CUDA_SUCCESS) {
    *pending = true;
    if (e)
      *e = "dlib table upload synchronization failed";
    return false;
  }
  return true;
}
} // namespace
struct PreparedCudaDlibShapePredictor::Impl {
  PreparedCudaDlibSetup setup;
  ValidatedModelArtifactIdentity artifact;
  cuda::kernels::PreparedDlibShapePredictorKernel kernel;
  std::array<Buffer, 9> buffers{};
  cuda::CudaTensor shape, features, output;
  bool ready = false, pending = false, stopped = false;
  bool setup_quarantined = false;
  std::uint64_t sequence = 0;
  ~Impl() {
    if (ready && !stopped)
      return;
    if (setup.binding.driver) {
      std::string e;
      if (kernel.module)
        (void)cuda::kernels::DestroyDlibShapePredictorKernel(&kernel, &e);
      for (auto &b : buffers)
        if (b.ptr)
          (void)setup.binding.driver->Free(b.ptr, &e);
      (void)shape.Free(setup.binding.driver, &e);
      (void)features.Free(setup.binding.driver, &e);
      (void)output.Free(setup.binding.driver, &e);
    }
  }
};
PreparedCudaDlibShapePredictor::PreparedCudaDlibShapePredictor()
    : impl_(std::make_unique<Impl>()) {}
PreparedCudaDlibShapePredictor::~PreparedCudaDlibShapePredictor() {
  std::string e;
  if (impl_ && !Shutdown(&e))
    (void)impl_.release();
}
bool PreparedCudaDlibShapePredictor::Prepare(const PreparedCudaDlibSetup &s,
                                             std::string *e) {
  if (e)
    e->clear();
  if (!impl_ || impl_->ready || impl_->setup_quarantined || !s.binding.driver ||
      !s.binding.context.Valid() || !s.binding.owner_context ||
      !s.binding.stream || !s.binding.stream_identity ||
      !s.binding.owner_identity || !s.binding.configuration_generation) {
    if (e)
      *e = "invalid prepared dlib setup";
    return false;
  }
#if !STUDIOCAST_HAVE_DLIB
  (void)s;
  if (e)
    *e = "dlib is unavailable in this build";
  return false;
#else
  auto registry = s.models_root.empty()
                      ? ModelPackRegistry::ScanDefault()
                      : ModelPackRegistry::Scan(s.models_root);
  auto id = s.model_id.empty()
                ? registry.DefaultModelIdForTask("face_landmarks")
                : s.model_id;
  auto pack = registry.Find("face_landmarks", id);
  if (!pack) {
    if (e)
      *e = "dlib landmark pack unavailable";
    return false;
  }
  const ModelFile *file = nullptr;
  for (const auto &f : pack->files)
    if (f.kind == "dlib_shape_predictor" && f.role == "main") {
      file = &f;
      break;
    }
  if (!file) {
    if (e)
      *e = "dlib landmark artifact unavailable";
    return false;
  }
  auto validation = ValidateModelArtifactIdentity(*pack, *file);
  if (!validation.ok()) {
    if (e)
      *e = "dlib landmark artifact identity mismatch";
    return false;
  }
  dlib::matrix<float, 0, 1> initial;
  std::vector<std::vector<dlib::impl::regression_tree>> forests;
  std::vector<std::vector<unsigned long>> anchors;
  std::vector<std::vector<dlib::vector<float, 2>>> deltas;
  try {
    std::ifstream wire(file->path, std::ios::binary);
    if (!ValidatePredictorWire(wire, s.resource_budget_bytes, e))
      return false;
    std::ifstream in(file->path, std::ios::binary);
    int version = 0;
    dlib::deserialize(version, in);
    if (version != 1)
      throw dlib::serialization_error("version");
    dlib::deserialize(initial, in);
    dlib::deserialize(forests, in);
    dlib::deserialize(anchors, in);
    dlib::deserialize(deltas, in);
    if (in.peek() != std::char_traits<char>::eof())
      throw dlib::serialization_error("trailing bytes");
  } catch (const std::exception &x) {
    if (e)
      *e = std::string("invalid dlib predictor: ") + x.what();
    return false;
  }
  if (initial.size() != 136 || forests.empty() ||
      anchors.size() != forests.size() || deltas.size() != forests.size()) {
    if (e)
      *e = "invalid dlib predictor dimensions";
    return false;
  }
  std::vector<float> iv(initial.begin(), initial.end()), lv, dv;
  std::vector<unsigned> ct{0}, ts{0}, tl{0}, cf{0}, av;
  struct S {
    unsigned a, b;
    float t;
  };
  std::vector<S> sv;
  std::size_t maxf = 0;
  auto finite = [](float x) { return std::isfinite(x); };
  for (float x : iv)
    if (!finite(x)) {
      if (e)
        *e = "non-finite dlib initial shape";
      return false;
    }
  for (std::size_t c = 0; c < forests.size(); ++c) {
    if (anchors[c].size() != deltas[c].size() || anchors[c].empty()) {
      if (e)
        *e = "invalid dlib cascade features";
      return false;
    }
    maxf = std::max(maxf, anchors[c].size());
    for (std::size_t i = 0; i < anchors[c].size(); ++i) {
      if (anchors[c][i] >= 68 || !finite(deltas[c][i].x()) ||
          !finite(deltas[c][i].y())) {
        if (e)
          *e = "invalid dlib anchor";
        return false;
      }
      av.push_back((unsigned)anchors[c][i]);
      dv.push_back(deltas[c][i].x());
      dv.push_back(deltas[c][i].y());
    }
    cf.push_back((unsigned)av.size());
    for (const auto &t : forests[c]) {
      if (t.leaf_values.size() != t.splits.size() + 1 ||
          t.leaf_values.empty() ||
          (t.leaf_values.size() & (t.leaf_values.size() - 1))) {
        if (e)
          *e = "incomplete dlib regression tree";
        return false;
      }
      for (const auto &x : t.splits) {
        if (x.idx1 >= anchors[c].size() || x.idx2 >= anchors[c].size() ||
            !finite(x.thresh)) {
          if (e)
            *e = "invalid dlib split";
          return false;
        }
        sv.push_back({(unsigned)x.idx1, (unsigned)x.idx2, x.thresh});
      }
      ts.push_back((unsigned)sv.size());
      for (const auto &leaf : t.leaf_values) {
        if (leaf.size() != 136) {
          if (e)
            *e = "invalid dlib leaf dimensions";
          return false;
        }
        for (float x : leaf) {
          if (!finite(x)) {
            if (e)
              *e = "non-finite dlib leaf";
            return false;
          }
          lv.push_back(x);
        }
      }
      tl.push_back((unsigned)(lv.size() / 136));
    }
    ct.push_back((unsigned)(ts.size() - 1));
  }
  std::size_t bytes = iv.size() * 4 + ct.size() * 4 + ts.size() * 4 +
                      tl.size() * 4 + sv.size() * sizeof(S) + lv.size() * 4 +
                      cf.size() * 4 + av.size() * 4 + dv.size() * 4 +
                      (136 + maxf + 136) * 4;
  constexpr std::size_t allocation_row = 8192;
  constexpr std::size_t maximum_padding = 9 * (allocation_row - 1);
  if (bytes <= std::numeric_limits<std::size_t>::max() - maximum_padding)
    bytes += maximum_padding;
  else
    bytes = std::numeric_limits<std::size_t>::max();
  if (bytes > s.resource_budget_bytes) {
    if (e)
      *e = "dlib prepared resource budget exceeded";
    return false;
  }
  auto candidate = std::make_unique<Impl>();
  candidate->setup = s;
  candidate->artifact = validation.identity;
  const void *ptrs[] = {iv.data(), ct.data(), ts.data(), tl.data(), sv.data(),
                        lv.data(), cf.data(), av.data(), dv.data()};
  std::size_t sizes[] = {iv.size() * 4, ct.size() * 4,         ts.size() * 4,
                         tl.size() * 4, sv.size() * sizeof(S), lv.size() * 4,
                         cf.size() * 4, av.size() * 4,         dv.size() * 4};
  for (int i = 0; i < 9; ++i)
    if (!AllocUpload(s.binding.driver, s.binding.stream, ptrs[i], sizes[i],
                     &candidate->buffers[i], &candidate->pending, e)) {
      if (e)
        *e = "dlib table " + std::to_string(i) + ": " + *e;
      if (candidate->pending) {
        candidate->setup_quarantined = true;
        impl_ = std::move(candidate);
      }
      return false;
    }
  if (!candidate->shape.AllocateNchwF32(s.binding.driver, 1, 1, 1, 136, e) ||
      !candidate->features.AllocateNchwF32(s.binding.driver, 1, 1, 1, (int)maxf,
                                           e) ||
      !candidate->output.AllocateNchwF32(s.binding.driver, 1, 1, 1, 136, e))
    return false;
  if (!cuda::kernels::PrepareDlibShapePredictorKernel(
          *s.binding.driver, s.binding.owner_context, s.binding.stream,
          &candidate->kernel, e)) {
    if (candidate->kernel.module) {
      candidate->setup_quarantined = true;
      impl_ = std::move(candidate);
    }
    return false;
  }
  if (s.binding.driver->f().cuStreamSynchronize(s.binding.stream) !=
      cuda::CUDA_SUCCESS) {
    if (e)
      *e = "dlib setup upload synchronization failed";
    return false;
  }
  candidate->ready = true;
  impl_ = std::move(candidate);
  return true;
#endif
}
PreparedCudaDlibResult PreparedCudaDlibShapePredictor::Execute(
    const PreparedCudaDlibBinding &b, const cuda::CudaImage &im,
    const PreparedCudaFaceBox &box, std::uint64_t seq,
    std::string *e) noexcept {
  PreparedCudaDlibResult r;
  if (e)
    e->clear();
  if (!impl_ || !impl_->ready || impl_->setup_quarantined || impl_->stopped) {
    r.status = PreparedCudaDlibStatus::shutdown;
    return r;
  }
  if (!Same(b, impl_->setup.binding) || !seq || seq <= impl_->sequence) {
    r.status = PreparedCudaDlibStatus::stale_binding;
    return r;
  }
  impl_->sequence = seq;
  auto &p = impl_->buffers;
  if (!cuda::kernels::LaunchDlibShapePredictor(
          impl_->kernel, im, box.left, box.top, box.right, box.bottom, p[0].ptr,
          p[1].ptr, p[2].ptr, p[3].ptr, p[4].ptr, p[5].ptr, p[6].ptr, p[7].ptr,
          p[8].ptr, (unsigned)(p[1].bytes / 4 - 1), impl_->shape.ptr,
          impl_->features.ptr, impl_->output.ptr, e)) {
    r.status = PreparedCudaDlibStatus::launch_failed;
    return r;
  }
  impl_->pending = true;
  r.pending_work = true;
  r.status = PreparedCudaDlibStatus::success;
  return r;
}
bool PreparedCudaDlibShapePredictor::Shutdown(std::string *e) noexcept {
  if (e)
    e->clear();
  if (!impl_ || impl_->stopped)
    return true;
  if (!impl_->ready && !impl_->setup_quarantined) {
    impl_->stopped = true;
    return true;
  }
  if (impl_->pending) {
    if (impl_->setup.binding.driver->f().cuStreamSynchronize(
            impl_->setup.binding.stream) != cuda::CUDA_SUCCESS) {
      if (e)
        *e = "dlib CUDA work is not quiescent; retry shutdown";
      return false;
    }
    impl_->pending = false;
  }
  if (!cuda::kernels::DestroyDlibShapePredictorKernel(&impl_->kernel, e))
    return false;
  std::string x;
  for (auto &b : impl_->buffers)
    if (b.ptr) {
      if (!impl_->setup.binding.driver->Free(b.ptr, &x)) {
        if (e)
          *e = x;
        return false;
      }
      b.ptr = 0;
    }
  if (!impl_->shape.Free(impl_->setup.binding.driver, e) ||
      !impl_->features.Free(impl_->setup.binding.driver, e) ||
      !impl_->output.Free(impl_->setup.binding.driver, e))
    return false;
  impl_->ready = false;
  impl_->setup_quarantined = false;
  impl_->stopped = true;
  return true;
}
bool PreparedCudaDlibShapePredictor::ready() const noexcept {
  return impl_ && impl_->ready && !impl_->stopped;
}
const ValidatedModelArtifactIdentity &
PreparedCudaDlibShapePredictor::artifact() const noexcept {
  return impl_->artifact;
}
const cuda::CudaTensor &
PreparedCudaDlibShapePredictor::resident_landmarks() const noexcept {
  return impl_->output;
}
} // namespace studiocast::open_video
