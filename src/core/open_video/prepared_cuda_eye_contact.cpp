#include "core/open_video/prepared_cuda_eye_contact.h"
#include "core/open_video/native_prepared_cuda_eye_contact_runtime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace studiocast::open_video {
PreparedCudaEyeContactStatus
ValidateResidentDlibLandmarks(const PreparedCudaDlibResult &result,
                              const cuda::CudaTensor &tensor) noexcept {
  if (result.status != PreparedCudaDlibStatus::success)
    return result.pending_work
               ? PreparedCudaEyeContactStatus::synchronization_failed
               : PreparedCudaEyeContactStatus::landmark_failed;
  return tensor.Valid() && tensor.n == 1 && tensor.c == 1 && tensor.h == 1 &&
                 tensor.w == 136
             ? PreparedCudaEyeContactStatus::executed
             : PreparedCudaEyeContactStatus::graph_mismatch;
}
namespace {
std::atomic<std::uint64_t> g_next_eye_handle{1};

bool SameBinding(const PreparedCudaEyeContactBinding &a,
                 const PreparedCudaEyeContactBinding &b) noexcept {
  return a.driver == b.driver && a.context == b.context &&
         a.owner_context == b.owner_context && a.stream == b.stream &&
         a.stream_identity == b.stream_identity &&
         a.owner_identity == b.owner_identity &&
         a.configuration_generation == b.configuration_generation &&
         a.model_generation == b.model_generation;
}

class PreparedAnalysis final : public IPreparedCudaEyeContactAnalysis {
public:
  PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactSetup &s) noexcept override {
    std::string error;
    borrowed_ = s.shared_yunet != nullptr;
    if (borrowed_) {
      borrowed_binding_ = *s.shared_yunet;
      active_yunet_ = borrowed_binding_.producer;
      if (!active_yunet_ || borrowed_binding_.cache != s.analysis_cache ||
          borrowed_binding_.binding.driver != s.binding.driver ||
          borrowed_binding_.binding.context != s.binding.context ||
          borrowed_binding_.binding.stream != s.binding.stream ||
          borrowed_binding_.binding.stream_identity !=
              s.binding.stream_identity ||
          borrowed_binding_.binding.owner_identity !=
              s.binding.owner_identity ||
          borrowed_binding_.binding.configuration_generation !=
              s.binding.configuration_generation ||
          borrowed_binding_.frame_width != s.frame_width ||
          borrowed_binding_.frame_height != s.frame_height ||
          borrowed_binding_.artifact != s.models.yunet ||
          !active_yunet_->MatchesPreparedBinding(borrowed_binding_))
        return PreparedCudaEyeContactStatus::model_invalid;
    } else {
      if (!yunet_.Prepare(s.yunet_setup, &error))
        return PreparedCudaEyeContactStatus::model_invalid;
      if (!yunet_.PrepareCache(s.analysis_cache, &error))
        return PreparedCudaEyeContactStatus::resource_exhausted;
      active_yunet_ = &yunet_;
    }
    if (!dlib_.Prepare(s.dlib_setup, &error))
      return PreparedCudaEyeContactStatus::model_invalid;
    return PreparedCudaEyeContactStatus::executed;
  }
  const ValidatedModelArtifactIdentity &artifact() const noexcept override {
    return active_yunet_ ? active_yunet_->artifact() : yunet_.artifact();
  }
  const ValidatedModelArtifactIdentity &
  landmark_artifact() const noexcept override {
    return dlib_.artifact();
  }
  PreparedCudaEyeContactAnalysisResult
  Detect(const PreparedCudaEyeContactBinding &b, const cuda::CudaImage &image,
         std::uint64_t sequence, FrameAnalysisCache *cache,
         const cuda::CudaPreparedTransferCompletion::WaitPolicy &wait) noexcept
      override {
    PreparedCudaEyeContactAnalysisResult out;
    PreparedCudaYunetBinding yb{b.driver,         b.context,
                                b.stream,         b.stream_identity,
                                b.owner_identity, b.configuration_generation};
    if (!active_yunet_ || (borrowed_ && !active_yunet_->MatchesPreparedBinding(
                                            borrowed_binding_)))
      return out;
    const auto result =
        active_yunet_->Execute(yb, image, sequence, cache, wait);
    out.enqueued_readbacks = result.enqueued_readbacks;
    out.completed_readbacks = result.completed_readbacks;
    out.completion_waits = result.completion_waits;
    out.cpu_tail_boundaries = result.cpu_tail_boundaries;
    out.pending_work = result.pending_work;
    out.device_lost = result.device_lost;
    switch (result.status) {
    case PreparedCudaYunetStatus::success_faces:
    case PreparedCudaYunetStatus::success_no_faces:
      out.status = PreparedCudaEyeContactStatus::executed;
      break;
    case PreparedCudaYunetStatus::timeout:
      out.status = PreparedCudaEyeContactStatus::timeout;
      break;
    case PreparedCudaYunetStatus::stop_requested:
      out.status = PreparedCudaEyeContactStatus::stop_requested;
      break;
    case PreparedCudaYunetStatus::cancelled:
      out.status = PreparedCudaEyeContactStatus::cancelled;
      break;
    case PreparedCudaYunetStatus::device_lost:
      out.status = PreparedCudaEyeContactStatus::device_lost;
      break;
    default:
      out.status = PreparedCudaEyeContactStatus::analysis_failed;
      break;
    }
    if (out.status != PreparedCudaEyeContactStatus::executed || !cache ||
        !cache->face_detections)
      return out;
    const FaceDetection *best = nullptr;
    for (const auto &face : *cache->face_detections) {
      if (!best || face.score > best->score + 1e-5F ||
          (std::abs(face.score - best->score) < 1e-5F &&
           face.w * face.h > best->w * best->h))
        best = &face;
    }
    if (best) {
      out.found = true;
      out.face = {best->x, best->y, best->x + best->w, best->y + best->h};
    }
    return out;
  }
  PreparedCudaEyeContactStatus
  Landmarks(const PreparedCudaEyeContactBinding &b,
            const cuda::CudaImage &image, const PreparedCudaFaceBox &face,
            std::uint64_t sequence,
            const cuda::CudaTensor **resident) noexcept override {
    if (resident)
      *resident = nullptr;
    PreparedCudaDlibBinding db{b.driver,
                               b.context,
                               b.owner_context,
                               b.stream,
                               b.stream_identity,
                               b.owner_identity,
                               b.configuration_generation};
    const auto result = dlib_.Execute(db, image, face, sequence, nullptr);
    const auto &tensor = dlib_.resident_landmarks();
    const auto status = ValidateResidentDlibLandmarks(result, tensor);
    if (status != PreparedCudaEyeContactStatus::executed)
      return status;
    if (resident)
      *resident = &tensor;
    return PreparedCudaEyeContactStatus::executed;
  }
  PreparedCudaEyeContactStatus Shutdown() noexcept override {
    std::string error;
    const bool d = dlib_.Shutdown(&error);
    const bool y = borrowed_ || yunet_.Shutdown(&error);
    return d && y ? PreparedCudaEyeContactStatus::shutdown
                  : PreparedCudaEyeContactStatus::shutdown_failed;
  }

private:
  PreparedCudaYunetDetector yunet_;
  PreparedCudaYunetDetector *active_yunet_ = nullptr;
  PreparedCudaYunetPreparedBinding borrowed_binding_{};
  bool borrowed_ = false;
  PreparedCudaDlibShapePredictor dlib_;
};

bool ValidSetup(const PreparedCudaEyeContactSetup &s) noexcept {
  return s.binding.driver && s.binding.context.Valid() &&
         s.binding.owner_context && s.binding.stream &&
         s.binding.stream_identity && s.binding.owner_identity &&
         s.binding.configuration_generation && s.binding.model_generation &&
         s.frame_width > 0 && s.frame_height > 0 && s.resource_budget_bytes &&
         s.analysis_cache && s.models.yunet.Valid() &&
         s.models.landmarks.Valid() && s.models.left.Valid() &&
         s.models.right.Valid() && s.models.eye_pack && s.models.left_file &&
         s.models.right_file && !s.models.left_path.empty() &&
         !s.models.right_path.empty();
}

bool CheckedAdd(std::size_t amount, std::size_t *total) noexcept {
  if (!total || amount > std::numeric_limits<std::size_t>::max() - *total)
    return false;
  *total += amount;
  return true;
}
} // namespace

struct PreparedCudaEyeContact::Impl {
  std::unique_ptr<IPreparedCudaEyeContactAnalysis> owned_analysis;
  std::unique_ptr<IPreparedCudaEyeContactRuntime> owned_runtime;
  IPreparedCudaEyeContactAnalysis *analysis = nullptr;
  IPreparedCudaEyeContactRuntime *runtime = nullptr;
  PreparedCudaEyeContactSetup setup{};
  PreparedCudaEyeContactHandle handle{};
  bool ready = false;
  bool cleanup_pending = false;
};

PreparedCudaEyeContact::PreparedCudaEyeContact()
    : impl_(std::make_unique<Impl>()) {
  impl_->owned_analysis = std::make_unique<PreparedAnalysis>();
  impl_->owned_runtime =
      std::make_unique<NativePreparedCudaEyeContactRuntime>();
  impl_->analysis = impl_->owned_analysis.get();
  impl_->runtime = impl_->owned_runtime.get();
}
PreparedCudaEyeContact::PreparedCudaEyeContact(
    IPreparedCudaEyeContactAnalysis *analysis,
    IPreparedCudaEyeContactRuntime *runtime) noexcept
    : impl_(std::make_unique<Impl>()) {
  impl_->analysis = analysis;
  impl_->runtime = runtime;
}
PreparedCudaEyeContact::PreparedCudaEyeContact(
    std::unique_ptr<IPreparedCudaEyeContactAnalysis> analysis,
    std::unique_ptr<IPreparedCudaEyeContactRuntime> runtime) noexcept
    : impl_(std::make_unique<Impl>()) {
  impl_->owned_analysis = std::move(analysis);
  impl_->owned_runtime = std::move(runtime);
  impl_->analysis = impl_->owned_analysis.get();
  impl_->runtime = impl_->owned_runtime.get();
}
PreparedCudaEyeContact::~PreparedCudaEyeContact() {
  if (impl_ && !Shutdown(nullptr))
    // Pending CUDA work still owns the sessions, bindings, modules, and
    // buffers reachable through Impl. The borrowed driver/context must outlive
    // successful Shutdown; when it does not, leaking is the only safe teardown.
    (void)impl_.release();
}

bool PreparedCudaEyeContact::Prepare(const PreparedCudaEyeContactSetup &s,
                                     PreparedCudaEyeContactHandle *handle,
                                     std::string *error) {
  if (error)
    error->clear();
  if (handle)
    *handle = {};
  if (!impl_ || !impl_->analysis || !impl_->runtime || !ValidSetup(s)) {
    if (error)
      *error = "invalid prepared Eye Contact setup";
    return false;
  }
  if ((impl_->ready || impl_->cleanup_pending) && !Shutdown(error))
    return false;
  auto frozen = s;
  std::size_t accounted = 0;
  if ((!s.shared_yunet &&
       !CheckedAdd(s.yunet_setup.resource_budget_bytes, &accounted)) ||
      !CheckedAdd(s.dlib_setup.resource_budget_bytes, &accounted) ||
      accounted > s.resource_budget_bytes) {
    if (error)
      *error = "prepared Eye Contact aggregate resource budget exceeded";
    return false;
  }
  frozen.yunet_setup.binding = {
      s.binding.driver,         s.binding.context,
      s.binding.stream,         s.binding.stream_identity,
      s.binding.owner_identity, s.binding.configuration_generation};
  frozen.yunet_setup.cuda_device_id = s.cuda_device_id;
  frozen.yunet_setup.frame_width = s.frame_width;
  frozen.yunet_setup.frame_height = s.frame_height;
  frozen.dlib_setup.binding = {s.binding.driver,
                               s.binding.context,
                               s.binding.owner_context,
                               s.binding.stream,
                               s.binding.stream_identity,
                               s.binding.owner_identity,
                               s.binding.configuration_generation};
  auto status = impl_->analysis->Prepare(frozen);
  if (status == PreparedCudaEyeContactStatus::executed &&
      (!(impl_->analysis->artifact() == s.models.yunet) ||
       !(impl_->analysis->landmark_artifact() == s.models.landmarks)))
    status = PreparedCudaEyeContactStatus::model_invalid;
  if (status == PreparedCudaEyeContactStatus::executed) {
    PreparedCudaEyeContactRuntimeConfig config{frozen.binding,
                                               frozen.models,
                                               frozen.cuda_device_id,
                                               frozen.frame_width,
                                               frozen.frame_height,
                                               frozen.resource_budget_bytes -
                                                   accounted,
                                               frozen.wait};
    status = impl_->runtime->Prepare(config);
  }
  if (status != PreparedCudaEyeContactStatus::executed) {
    const auto runtime_shutdown = impl_->runtime->Shutdown();
    const auto analysis_shutdown = impl_->analysis->Shutdown();
    impl_->cleanup_pending =
        runtime_shutdown == PreparedCudaEyeContactStatus::shutdown_failed ||
        analysis_shutdown == PreparedCudaEyeContactStatus::shutdown_failed;
    if (error)
      *error = "prepared Eye Contact setup failed";
    return false;
  }
  impl_->setup = std::move(frozen);
  impl_->handle = {g_next_eye_handle.fetch_add(1),
                   s.binding.configuration_generation,
                   s.binding.model_generation};
  impl_->ready = true;
  impl_->cleanup_pending = false;
  if (handle)
    *handle = impl_->handle;
  return true;
}

PreparedCudaEyeContactFrameResult
PreparedCudaEyeContact::Execute(const PreparedCudaEyeContactHandle &h,
                                const PreparedCudaEyeContactBinding &binding,
                                std::uint64_t sequence,
                                const cuda::CudaImage &input) noexcept {
  PreparedCudaEyeContactFrameResult out;
  out.output = &input;
  if (!impl_ || !impl_->ready) {
    out.status = PreparedCudaEyeContactStatus::shutdown;
    return out;
  }
  if (h.id != impl_->handle.id ||
      h.configuration_generation != impl_->handle.configuration_generation ||
      h.model_generation != impl_->handle.model_generation ||
      !SameBinding(binding, impl_->setup.binding) ||
      input.w != impl_->setup.frame_width ||
      input.h != impl_->setup.frame_height ||
      input.format != cuda::PixelFormatGpu::rgb_u8) {
    return out;
  }
  if (impl_->setup.strength <= 0) {
    out.status = PreparedCudaEyeContactStatus::preserved_input;
    out.valid_input_preserved = true;
    return out;
  }
  // The published look-away switch has no behavior; retain that contract.
  (void)impl_->setup.look_away_enabled;
  auto analysis = impl_->analysis->Detect(
      binding, input, sequence, impl_->setup.analysis_cache, impl_->setup.wait);
  out.enqueued_readbacks = analysis.enqueued_readbacks;
  out.completed_readbacks = analysis.completed_readbacks;
  out.completion_waits = analysis.completion_waits;
  out.cpu_tail_boundaries = analysis.cpu_tail_boundaries;
  out.pending_work = analysis.pending_work;
  out.device_lost = analysis.device_lost;
  if (analysis.status != PreparedCudaEyeContactStatus::executed) {
    out.status = analysis.status;
    out.valid_input_preserved = !analysis.pending_work && !analysis.device_lost;
    if (!out.valid_input_preserved)
      out.output = nullptr;
    return out;
  }
  out.face_found = analysis.found;
  if (!analysis.found) {
    out.status = PreparedCudaEyeContactStatus::preserved_input;
    out.valid_input_preserved = true;
    return out;
  }
  const cuda::CudaTensor *landmarks = nullptr;
  out.status = impl_->analysis->Landmarks(binding, input, analysis.face,
                                          sequence, &landmarks);
  if (out.status != PreparedCudaEyeContactStatus::executed || !landmarks) {
    out.valid_input_preserved =
        out.status != PreparedCudaEyeContactStatus::device_lost &&
        out.status != PreparedCudaEyeContactStatus::synchronization_failed;
    if (!out.valid_input_preserved)
      out.output = nullptr;
    return out;
  }
  auto runtime = impl_->runtime->Execute(
      binding, input, *landmarks,
      static_cast<float>(std::clamp(impl_->setup.strength, 0, 100)) / 100.0F);
  out.status = runtime.status;
  out.output = runtime.output;
  out.pending_work = runtime.pending_work;
  out.device_lost = runtime.device_lost;
  out.inference_attempts = runtime.inference_attempts;
  out.inference_successes = runtime.inference_successes;
  if (runtime.output == &input && !runtime.pending_work &&
      !runtime.device_lost) {
    out.output = &input;
    out.valid_input_preserved = true;
  }
  return out;
}

bool PreparedCudaEyeContact::Shutdown(std::string *error) noexcept {
  if (error)
    error->clear();
  if (!impl_ || (!impl_->ready && !impl_->cleanup_pending))
    return true;
  const auto runtime = impl_->runtime->Shutdown();
  const auto analysis = impl_->analysis->Shutdown();
  if (runtime == PreparedCudaEyeContactStatus::shutdown_failed ||
      analysis == PreparedCudaEyeContactStatus::shutdown_failed) {
    impl_->cleanup_pending = true;
    if (error)
      *error = "prepared Eye Contact shutdown remains pending";
    return false;
  }
  impl_->ready = false;
  impl_->cleanup_pending = false;
  return true;
}
bool PreparedCudaEyeContact::ready() const noexcept {
  return impl_ && impl_->ready;
}
} // namespace studiocast::open_video
