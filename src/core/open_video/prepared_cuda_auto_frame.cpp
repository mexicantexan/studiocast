#include "core/open_video/prepared_cuda_auto_frame.h"

#include "core/cuda/kernels/resize_bilinear.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace studiocast::open_video {
namespace {

bool SameBinding(const PreparedCudaAutoFrameBinding &a,
                 const PreparedCudaAutoFrameBinding &b) noexcept {
  return a.driver == b.driver && a.context == b.context &&
         a.owner_context == b.owner_context && a.stream == b.stream &&
         a.stream_identity == b.stream_identity &&
         a.owner_identity == b.owner_identity &&
         a.configuration_generation == b.configuration_generation &&
         a.model_generation == b.model_generation;
}
PreparedCudaAutoFrameRect Clamp(PreparedCudaAutoFrameRect r, int w, int h) {
  r.w = std::clamp(r.w, 1.F, static_cast<float>(w));
  r.h = std::clamp(r.h, 1.F, static_cast<float>(h));
  r.x = std::clamp(r.x, 0.F, static_cast<float>(w) - r.w);
  r.y = std::clamp(r.y, 0.F, static_cast<float>(h) - r.h);
  return r;
}
PreparedCudaAutoFrameRect Lerp(const PreparedCudaAutoFrameRect &a,
                               const PreparedCudaAutoFrameRect &b, float t) {
  return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.w + (b.w - a.w) * t,
          a.h + (b.h - a.h) * t};
}
bool Usable(const PreparedCudaAutoFrameTrackingBox &b, int w, int h) {
  const auto &r = b.box;
  return b.source != PreparedCudaAutoFrameSource::none && std::isfinite(r.x) &&
         std::isfinite(r.y) && std::isfinite(r.w) && std::isfinite(r.h) &&
         r.w > .5F && r.h > .5F && r.x < w && r.y < h && r.x + r.w > 0 &&
         r.y + r.h > 0;
}
void Add(hal::TransferCounters *a, const hal::TransferCounters &b) {
  a->uploads += b.uploads;
  a->final_readbacks += b.final_readbacks;
  a->intermediate_readbacks += b.intermediate_readbacks;
  a->intermediate_uploads += b.intermediate_uploads;
  a->synchronizations += b.synchronizations;
  a->cpu_tail_boundaries += b.cpu_tail_boundaries;
}

class NativeCropRuntime final : public IPreparedCudaAutoFrameCropRuntime {
public:
  bool Prepare(const PreparedCudaAutoFrameBinding &b, int w, int h,
               std::size_t budget, std::string *error) noexcept override {
    if (live_ && !Shutdown(error))
      return false;
    if (!b.driver || !b.context.Valid() || !b.owner_context || !b.stream ||
        w <= 0 || h <= 0 ||
        static_cast<std::size_t>(w) > std::numeric_limits<std::size_t>::max() /
                                          3u / static_cast<std::size_t>(h) ||
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u >
            budget) {
      if (error)
        *error = "Auto Frame crop resource budget exhausted";
      return false;
    }
    if (b.driver->BindPreparedContext(b.context) != cuda::CUDA_SUCCESS) {
      if (error)
        *error = "Auto Frame owner context binding failed";
      return false;
    }
    if (!output_.Allocate(b.driver, w, h, cuda::PixelFormatGpu::rgb_u8, error))
      return false;
    binding_ = b;
    live_ = true;
    if (output_.pitch > budget / static_cast<std::size_t>(h)) {
      if (error)
        *error = "Auto Frame pitched output exceeds resource budget";
      return false;
    }
    if (!cuda::kernels::PrepareResizeBilinearKernel(
            *b.driver, b.owner_context, b.stream, &kernel_, error)) {
      return false;
    }
    return true;
  }
  bool Crop(const cuda::CudaImage &in, const PreparedCudaAutoFrameRect &r,
            const cuda::CudaImage **out, std::string *error) noexcept override {
    if (out)
      *out = nullptr;
    if (!live_ || !cuda::kernels::LaunchCropResizeBilinearPrepared(
                      kernel_, in, output_, r.x, r.y, r.w, r.h, error))
      return false;
    if (out)
      *out = &output_;
    return true;
  }
  bool Shutdown(std::string *error) noexcept override {
    if (!live_)
      return true;
    if (binding_.driver->BindPreparedContext(binding_.context) !=
            cuda::CUDA_SUCCESS ||
        !binding_.driver->StreamSynchronize(binding_.stream, error))
      return false;
    if (!cuda::kernels::DestroyResizeBilinearKernelPrepared(&kernel_, error))
      return false;
    if (!output_.Free(binding_.driver, error))
      return false;
    live_ = false;
    return true;
  }
  ~NativeCropRuntime() override { (void)Shutdown(nullptr); }

private:
  PreparedCudaAutoFrameBinding binding_{};
  cuda::kernels::PreparedResizeBilinearKernel kernel_{};
  cuda::CudaImage output_{};
  bool live_ = false;
};
} // namespace

PreparedCudaAutoFrameYunetAdapter::PreparedCudaAutoFrameYunetAdapter(
    PreparedCudaYunetDetector *d, PreparedCudaYunetBinding b) noexcept
    : detector_(d), binding_(b) {}
PreparedCudaAutoFrameAnalysisResult PreparedCudaAutoFrameYunetAdapter::Analyze(
    const PreparedCudaAutoFrameBinding &b, const cuda::CudaImage &image,
    std::uint64_t seq, FrameAnalysisCache *cache,
    const cuda::CudaPreparedTransferCompletion::WaitPolicy &wait) noexcept {
  PreparedCudaAutoFrameAnalysisResult out;
  if (!detector_ || b.driver != binding_.driver ||
      b.context != binding_.context || b.stream != binding_.stream ||
      b.stream_identity != binding_.stream_identity ||
      b.owner_identity != binding_.owner_identity ||
      b.configuration_generation != binding_.configuration_generation)
    return out;
  const auto r = detector_->Execute(binding_, image, seq, cache, wait);
  out.pending_work = r.pending_work;
  out.device_lost = r.device_lost;
  out.synchronization_failed =
      r.status == PreparedCudaYunetStatus::completion_failed ||
      r.status == PreparedCudaYunetStatus::timeout ||
      r.status == PreparedCudaYunetStatus::stop_requested ||
      r.status == PreparedCudaYunetStatus::cancelled;
  out.completion_waits = r.completion_waits;
  out.cpu_tail_boundaries = r.cpu_tail_boundaries;
  out.transfers.intermediate_readbacks = r.completed_readbacks;
  out.transfers.synchronizations = r.completion_waits;
  out.transfers.cpu_tail_boundaries = r.cpu_tail_boundaries;
  out.ok = r.status == PreparedCudaYunetStatus::success_faces ||
           r.status == PreparedCudaYunetStatus::success_no_faces;
  if (!out.ok || !cache || !cache->face_detections)
    return out;
  const FaceDetection *best = nullptr;
  for (const auto &f : *cache->face_detections)
    if (!best || f.w * f.h > best->w * best->h)
      best = &f;
  if (best) {
    out.found = true;
    out.box.source = PreparedCudaAutoFrameSource::face;
    out.box.box = {best->x - best->w * .4F, best->y - best->h * .4F,
                   best->w * 1.8F, best->h * 2.6F};
    out.box.confidence = best->score;
  }
  return out;
}

PreparedCudaAutoFrameMatteAdapter::PreparedCudaAutoFrameMatteAdapter(
    void *owner, PreparedCudaAutoFrameMatteProducer producer,
    FrameMatteArtifactKey key, int frame_width, int frame_height) noexcept
    : owner_(owner), producer_(producer), key_(std::move(key)),
      frame_width_(frame_width), frame_height_(frame_height) {}
PreparedCudaAutoFrameAnalysisResult PreparedCudaAutoFrameMatteAdapter::Analyze(
    const PreparedCudaAutoFrameBinding &b, const cuda::CudaImage &i,
    std::uint64_t s, FrameAnalysisCache *c,
    const cuda::CudaPreparedTransferCompletion::WaitPolicy &w) noexcept {
  if (c) {
    if (const auto *artifact = c->artifacts.FindMatte(s, key_);
        artifact && key_.storage == FrameMatteStorage::cpu_f32_alpha &&
        artifact->cpu_alpha.size() ==
            static_cast<std::size_t>(key_.matte_width) * key_.matte_height) {
      PreparedCudaAutoFrameAnalysisResult result;
      result.ok = true;
      result.found = ComputeBox(artifact->cpu_alpha.data(), key_.matte_width,
                                key_.matte_height, frame_width_, frame_height_,
                                &result.box);
      result.cpu_tail_boundaries = 1;
      result.transfers.cpu_tail_boundaries = 1;
      return result;
    }
  }
  return producer_ ? producer_(owner_, b, i, s, c, w)
                   : PreparedCudaAutoFrameAnalysisResult{};
}
bool PreparedCudaAutoFrameMatteAdapter::HasSameFrameArtifact(
    std::uint64_t sequence, const FrameAnalysisCache *cache) const noexcept {
  return cache && cache->artifacts.FindMatte(sequence, key_) != nullptr;
}
bool PreparedCudaAutoFrameMatteAdapter::ComputeBox(
    const float *a, int aw, int ah, int fw, int fh,
    PreparedCudaAutoFrameTrackingBox *out) noexcept {
  if (out)
    *out = {};
  if (!a || !out || aw <= 0 || ah <= 0 || fw <= 0 || fh <= 0)
    return false;
  int x0 = aw, y0 = ah, x1 = -1, y1 = -1, count = 0;
  for (int y = 0; y < ah; ++y)
    for (int x = 0; x < aw; ++x)
      if (a[y * aw + x] > .35F) {
        x0 = std::min(x0, x);
        y0 = std::min(y0, y);
        x1 = std::max(x1, x);
        y1 = std::max(y1, y);
        ++count;
      }
  if (count < std::max(32, aw * ah / 500) || x1 < x0)
    return false;
  const float sx = static_cast<float>(fw) / aw,
              sy = static_cast<float>(fh) / ah;
  out->source = PreparedCudaAutoFrameSource::matte;
  out->box = {x0 * sx, y0 * sy, (x1 + 1 - x0) * sx, (y1 + 1 - y0) * sy};
  out->confidence = 1;
  return true;
}

struct PreparedCudaAutoFrame::Impl {
  std::unique_ptr<IPreparedCudaAutoFrameCropRuntime> owned;
  IPreparedCudaAutoFrameCropRuntime *crop = nullptr;
  PreparedCudaAutoFrameSetup setup{};
  PreparedCudaAutoFrameHandle handle{};
  PreparedCudaAutoFrameRect smooth{}, cached_matte{};
  PreparedCudaAutoFrameSource last_source = PreparedCudaAutoFrameSource::none;
  std::uint64_t last_seq = 0, last_matte_seq = 0;
  bool have_seq = false, have_smooth = false, have_matte = false, ready = false,
       stopped = false, cleanup_pending = false;
  std::uint64_t next_id = 1;
};
PreparedCudaAutoFrame::PreparedCudaAutoFrame()
    : impl_(std::make_unique<Impl>()) {
  impl_->owned = std::make_unique<NativeCropRuntime>();
  impl_->crop = impl_->owned.get();
}
PreparedCudaAutoFrame::PreparedCudaAutoFrame(
    IPreparedCudaAutoFrameCropRuntime *r) noexcept
    : impl_(std::make_unique<Impl>()) {
  impl_->crop = r;
}
PreparedCudaAutoFrame::~PreparedCudaAutoFrame() {
  if (impl_)
    (void)Shutdown(nullptr);
}

bool PreparedCudaAutoFrame::Prepare(const PreparedCudaAutoFrameSetup &s,
                                    PreparedCudaAutoFrameHandle *h,
                                    std::string *error) {
  if (error)
    error->clear();
  if (h)
    *h = {};
  if (!impl_ || !impl_->crop || !s.binding.driver ||
      !s.binding.context.Valid() || !s.binding.owner_context ||
      !s.binding.stream || s.frame_width <= 0 || s.frame_height <= 0 ||
      !s.analysis_cache || (!s.face && !s.matte) ||
      s.resource_budget_bytes == 0) {
    if (error)
      *error = "invalid prepared Auto Frame setup";
    return false;
  }
  if ((impl_->ready || impl_->cleanup_pending) && !Shutdown(error))
    return false;
  if (!impl_->crop->Prepare(s.binding, s.frame_width, s.frame_height,
                            s.resource_budget_bytes, error)) {
    const std::string primary_error = error ? *error : std::string{};
    impl_->cleanup_pending = true;
    std::string cleanup_error;
    if (impl_->crop->Shutdown(&cleanup_error))
      impl_->cleanup_pending = false;
    if (error) {
      *error = primary_error;
      if (impl_->cleanup_pending && !cleanup_error.empty())
        *error += "; cleanup pending: " + cleanup_error;
    }
    return false;
  }
  impl_->setup = s;
  impl_->handle = {impl_->next_id++, s.binding.configuration_generation,
                   s.binding.model_generation};
  impl_->last_seq = impl_->last_matte_seq = 0;
  impl_->have_seq = impl_->have_smooth = impl_->have_matte = false;
  impl_->last_source = PreparedCudaAutoFrameSource::none;
  impl_->ready = true;
  impl_->stopped = false;
  impl_->cleanup_pending = false;
  if (h)
    *h = impl_->handle;
  return true;
}

float PreparedCudaAutoFrame::SmoothingAlpha(int p) noexcept {
  return 1.F / (1.F + 9.F * (std::clamp(p, 0, 100) / 100.F));
}
PreparedCudaAutoFrameRect
PreparedCudaAutoFrame::CenterCrop(int w, int h, float aspect,
                                  float zoom) noexcept {
  zoom = std::max(1.F, zoom);
  float cw = w, ch = cw / aspect;
  if (ch > h) {
    ch = h;
    cw = ch * aspect;
  }
  cw = std::min<float>(w, cw / zoom);
  ch = std::min<float>(h, ch / zoom);
  return Clamp({(w - cw) / 2, (h - ch) / 2, cw, ch}, w, h);
}
PreparedCudaAutoFrameRect PreparedCudaAutoFrame::TargetCrop(
    const PreparedCudaAutoFrameTrackingBox &b, int w, int h, float aspect,
    const PreparedCudaAutoFrameKnobs &k) noexcept {
  const float strength = std::clamp(k.strength, 0, 100) / 100.F;
  const float pad = 2.4F - strength * 1.2F, cx = b.box.x + b.box.w * .5F;
  const float cy =
      b.box.y + b.box.h * .5F - std::clamp(k.headroom, 0.F, 1.F) * b.box.h;
  float ch = b.box.h * pad, cw = ch * aspect, minw = b.box.w * pad;
  if (cw < minw) {
    cw = minw;
    ch = cw / aspect;
  }
  cw = std::clamp(cw, w * .35F, float(w));
  ch = std::clamp(ch, h * .35F, float(h));
  return Clamp({cx - cw * .5F, cy - ch * .5F, cw, ch}, w, h);
}

PreparedCudaAutoFrameFrameResult PreparedCudaAutoFrame::Execute(
    const PreparedCudaAutoFrameHandle &h, const PreparedCudaAutoFrameBinding &b,
    std::uint64_t seq, const cuda::CudaImage &input) noexcept {
  PreparedCudaAutoFrameFrameResult out;
  out.output = &input;
  if (!impl_ || !impl_->ready || impl_->stopped) {
    out.status = PreparedCudaAutoFrameStatus::shutdown;
    return out;
  }
  if (h.id != impl_->handle.id ||
      h.configuration_generation != impl_->handle.configuration_generation ||
      h.model_generation != impl_->handle.model_generation ||
      !SameBinding(b, impl_->setup.binding) ||
      input.w != impl_->setup.frame_width ||
      input.h != impl_->setup.frame_height ||
      input.format != cuda::PixelFormatGpu::rgb_u8)
    return out;
  if (impl_->have_seq && seq != impl_->last_seq + 1) {
    impl_->have_smooth = impl_->have_matte = false;
    impl_->last_source = PreparedCudaAutoFrameSource::none;
    impl_->last_matte_seq = 0;
  }
  impl_->last_seq = seq;
  impl_->have_seq = true;
  PreparedCudaAutoFrameAnalysisResult ar;
  if (impl_->setup.face) {
    ar = impl_->setup.face->Analyze(b, input, seq, impl_->setup.analysis_cache,
                                    impl_->setup.wait);
    out.analysis_ran = true;
    Add(&out.transfers, ar.transfers);
    out.completion_waits += ar.completion_waits;
    out.cpu_tail_boundaries += ar.cpu_tail_boundaries;
    ar.transfers = {};
    ar.completion_waits = 0;
    ar.cpu_tail_boundaries = 0;
  }
  bool face_ok = impl_->setup.face && ar.ok;
  if (impl_->setup.face && !face_ok) {
    if (ar.device_lost)
      out.status = PreparedCudaAutoFrameStatus::device_lost;
    else if (ar.pending_work || ar.synchronization_failed)
      out.status = PreparedCudaAutoFrameStatus::synchronization_failed;
    else
      out.status = PreparedCudaAutoFrameStatus::preserved_input;
    out.valid_input_preserved =
        !ar.pending_work && !ar.device_lost && !ar.synchronization_failed;
    if (!out.valid_input_preserved)
      out.output = nullptr;
    return out;
  }
  if ((!impl_->setup.face || !ar.found) && impl_->setup.matte) {
    const bool shared = impl_->setup.matte->HasSameFrameArtifact(
        seq, impl_->setup.analysis_cache);
    const bool due = shared || impl_->last_matte_seq == 0 ||
                     seq >= impl_->last_matte_seq + 10;
    if (due) {
      ar = impl_->setup.matte->Analyze(
          b, input, seq, impl_->setup.analysis_cache, impl_->setup.wait);
      out.analysis_ran = true;
      if (ar.ok) {
        impl_->last_matte_seq = seq;
        impl_->have_matte = ar.found;
        if (ar.found)
          impl_->cached_matte = ar.box.box;
      }
    } else {
      ar.ok = true;
      ar.found = impl_->have_matte;
      ar.box.source = PreparedCudaAutoFrameSource::matte;
      ar.box.box = impl_->cached_matte;
    }
  }
  Add(&out.transfers, ar.transfers);
  out.completion_waits += ar.completion_waits;
  out.cpu_tail_boundaries += ar.cpu_tail_boundaries;
  if (!ar.ok || ar.pending_work) {
    if (ar.device_lost)
      out.status = PreparedCudaAutoFrameStatus::device_lost;
    else if (ar.pending_work || ar.synchronization_failed)
      out.status = PreparedCudaAutoFrameStatus::synchronization_failed;
    else
      out.status = PreparedCudaAutoFrameStatus::preserved_input;
    out.valid_input_preserved =
        !ar.pending_work && !ar.device_lost && !ar.synchronization_failed;
    if (!out.valid_input_preserved)
      out.output = nullptr;
    return out;
  }
  out.found = ar.found && Usable(ar.box, input.w, input.h);
  out.source = out.found ? ar.box.source : PreparedCudaAutoFrameSource::none;
  if (out.source != impl_->last_source) {
    impl_->have_smooth = false;
    impl_->last_source = out.source;
  }
  const float aspect =
      static_cast<float>(impl_->setup.frame_width) / impl_->setup.frame_height;
  auto target =
      out.found
          ? TargetCrop(ar.box, input.w, input.h, aspect, impl_->setup.knobs)
          : CenterCrop(input.w, input.h, aspect,
                       1.F + std::clamp(impl_->setup.knobs.strength, 0, 100) /
                                 100.F * .5F);
  impl_->smooth = impl_->have_smooth
                      ? Lerp(impl_->smooth, target,
                             SmoothingAlpha(impl_->setup.knobs.smoothing))
                      : target;
  impl_->smooth = Clamp(impl_->smooth, input.w, input.h);
  impl_->have_smooth = true;
  out.source_crop = impl_->smooth;
  const float tracked_x = out.found ? ar.box.box.x + ar.box.box.w * .5F
                                    : impl_->smooth.x + impl_->smooth.w * .5F;
  const float tracked_y = out.found ? ar.box.box.y + ar.box.box.h * .5F
                                    : impl_->smooth.y + impl_->smooth.h * .5F;
  out.tracked_center_x = (tracked_x - impl_->smooth.x) *
                         impl_->setup.frame_width / impl_->smooth.w;
  out.tracked_center_y = (tracked_y - impl_->smooth.y) *
                         impl_->setup.frame_height / impl_->smooth.h;
  const bool full = std::abs(impl_->smooth.x) <= .5F &&
                    std::abs(impl_->smooth.y) <= .5F &&
                    std::abs(impl_->smooth.w - input.w) <= .5F &&
                    std::abs(impl_->smooth.h - input.h) <= .5F;
  if (full) {
    out.status = PreparedCudaAutoFrameStatus::executed;
    out.valid_input_preserved = true;
    return out;
  }
  if (!impl_->crop->Crop(input, impl_->smooth, &out.output, nullptr)) {
    out.output = nullptr;
    out.status = PreparedCudaAutoFrameStatus::drop_frame;
    return out;
  }
  out.status = PreparedCudaAutoFrameStatus::executed;
  return out;
}
bool PreparedCudaAutoFrame::Shutdown(std::string *error) noexcept {
  if (!impl_ || (impl_->stopped && !impl_->cleanup_pending) ||
      (!impl_->ready && !impl_->cleanup_pending))
    return true;
  if (!impl_->crop->Shutdown(error))
    return false;
  impl_->ready = false;
  impl_->cleanup_pending = false;
  impl_->stopped = true;
  return true;
}
} // namespace studiocast::open_video
