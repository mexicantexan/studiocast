#include "core/maxine/production_resident_frame_executor.h"

#include "core/maxine/cuda_crop_scale.h"
#include "core/maxine/cuda_vignette.h"
#include "core/maxine/effects/ar_auto_frame_tracker.h"
#include "core/maxine/effects/ar_eye_contact_effect.h"
#include "core/maxine/effects/vfx_background_blur_effect.h"
#include "core/maxine/effects/vfx_denoise_effect.h"
#include "core/maxine/effects/vfx_green_screen_effect.h"
#include "core/maxine/effects/vfx_relighting_effect.h"
#include "core/maxine/effects/vfx_transfer_effect.h"
#include "core/video/convert.h"
#include "core/video/gpu_frame.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace studiocast::maxine {
namespace {

using Effects = video::effects::BroadcastCameraEffects;
using VirtualBackgroundMode = video::effects::VirtualBackgroundMode;
using effects::EffectExecutionTelemetry;

std::size_t TransferIndex(ProductionTransferKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

std::size_t StageIndex(ResidentStageKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

std::size_t CpuStageIndex(ProductionCpuStageKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

bool CheckedImageBytes(uint32_t width, uint32_t height,
                       std::size_t *bytes) noexcept {
  if (!bytes || width == 0 || height == 0)
    return false;
  constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
  if (width > max / 3u)
    return false;
  const std::size_t row = static_cast<std::size_t>(width) * 3u;
  if (height > max / row)
    return false;
  *bytes = row * static_cast<std::size_t>(height);
  return row <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool ParseHexByte(std::string_view value, uint8_t *output) noexcept {
  if (!output || value.size() != 2)
    return false;
  unsigned parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed, 16);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      parsed > 255)
    return false;
  *output = static_cast<uint8_t>(parsed);
  return true;
}

bool ParseRgbColor(std::string_view value,
                   std::array<uint8_t, 3> *bgr) noexcept {
  if (!bgr || value.size() != 7 || value.front() != '#')
    return false;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  if (!ParseHexByte(value.substr(1, 2), &r) ||
      !ParseHexByte(value.substr(3, 2), &g) ||
      !ParseHexByte(value.substr(5, 2), &b))
    return false;
  *bgr = {b, g, r};
  return true;
}

float Percent01(int value) noexcept {
  return static_cast<float>(std::clamp(value, 0, 100)) / 100.0f;
}

bool EffectsNeedVfx(const ProductionResidentSetup &setup) noexcept {
  constexpr uint32_t mask =
      ProductionResidentStageBit(ResidentStageKind::denoise) |
      ProductionResidentStageBit(ResidentStageKind::background_blur) |
      ProductionResidentStageBit(ResidentStageKind::background_remove) |
      ProductionResidentStageBit(ResidentStageKind::background_replace) |
      ProductionResidentStageBit(ResidentStageKind::relighting) |
      ProductionResidentStageBit(ResidentStageKind::transfer);
  return (setup.enabled_maxine_stage_mask & mask) != 0;
}

bool EffectsNeedAr(const ProductionResidentSetup &setup) noexcept {
  constexpr uint32_t mask =
      ProductionResidentStageBit(ResidentStageKind::eye_contact) |
      ProductionResidentStageBit(ResidentStageKind::auto_frame);
  return (setup.enabled_maxine_stage_mask & mask) != 0;
}

bool StageEnabled(const ProductionResidentSetup &setup,
                  ResidentStageKind kind) noexcept {
  return (setup.enabled_maxine_stage_mask & ProductionResidentStageBit(kind)) !=
         0;
}

bool SameResourceIdentity(const ProductionResidentSetup &a,
                          const ProductionResidentSetup &b) noexcept {
  return a.width == b.width && a.height == b.height &&
         a.runtime_identity == b.runtime_identity &&
         a.vfx_model_directory == b.vfx_model_directory;
}

bool SameEffectiveSetup(const ProductionResidentSetup &a,
                        const ProductionResidentSetup &b) noexcept {
  return SameResourceIdentity(a, b) &&
         a.configuration_generation == b.configuration_generation &&
         a.output_width == b.output_width &&
         a.output_height == b.output_height && a.effects == b.effects &&
         a.enabled_maxine_stage_mask == b.enabled_maxine_stage_mask &&
         a.vignette_center_x_px == b.vignette_center_x_px &&
         a.vignette_center_y_px == b.vignette_center_y_px &&
         a.replacement_background.width == b.replacement_background.width &&
         a.replacement_background.height == b.replacement_background.height &&
         a.replacement_background.asset_generation ==
             b.replacement_background.asset_generation;
}

class DefaultResidentDeviceOps final : public IResidentDeviceOps {
public:
  bool Initialize(CudaDriverApi *cuda, uintptr_t runtime_identity,
                  uint32_t required_ops_mask, std::string *error) override {
    if (runtime_identity_ != 0 && runtime_identity_ != runtime_identity) {
      crop_ = {};
      resize_ = {};
      vignette_ = {};
    }
    cuda_ = cuda;
    runtime_identity_ = runtime_identity;
    if (!cuda_ || !cuda_->IsInitialized()) {
      if (error)
        *error = "CUDA driver API is not initialized.";
      return false;
    }
    if ((required_ops_mask &
         ResidentDeviceOpBit(ResidentDeviceOp::crop_scale)) != 0 &&
        !crop_.Initialize(cuda_, error))
      return false;
    if ((required_ops_mask & ResidentDeviceOpBit(ResidentDeviceOp::resize)) !=
            0 &&
        !resize_.Initialize(cuda_, error))
      return false;
    if ((required_ops_mask & ResidentDeviceOpBit(ResidentDeviceOp::vignette)) !=
            0 &&
        !vignette_.Initialize(cuda_, error))
      return false;
    return true;
  }

  bool CropScale(const NvCVImage &source, NvCVImage *destination, float crop_x,
                 float crop_y, float crop_width, float crop_height,
                 CUstream stream, std::string *error) noexcept override {
    return crop_.CropScale(source, destination, crop_x, crop_y, crop_width,
                           crop_height, stream, error);
  }

  bool Resize(const NvCVImage &source, NvCVImage *destination, CUstream stream,
              std::string *error) noexcept override {
    return resize_.Resize(source, destination, stream, error);
  }

  bool Vignette(NvCVImage *image, float intensity, float center_x,
                float center_y, CUstream stream,
                std::string *error) noexcept override {
    return vignette_.ApplyInPlace(image, intensity, center_x, center_y, stream,
                                  error);
  }

  bool Synchronize(CUstream stream, std::string *error) noexcept override {
    return cuda_ && cuda_->StreamSynchronize(stream, error);
  }

private:
  CudaDriverApi *cuda_ = nullptr;
  uintptr_t runtime_identity_ = 0;
  CudaBgrCropScale crop_{};
  CudaBgrResizeBilinear resize_{};
  CudaBgrVignette vignette_{};
};

void AccumulateExecutionDelta(const EffectExecutionTelemetry &before,
                              const EffectExecutionTelemetry &after,
                              ProductionResidentTelemetry *telemetry) noexcept {
  if (!telemetry)
    return;
  telemetry->synchronous_sdk_runs.attempts +=
      after.synchronous_run_attempts - before.synchronous_run_attempts;
  telemetry->synchronous_sdk_runs.successes +=
      after.synchronous_run_successes - before.synchronous_run_successes;
  telemetry->asynchronous_sdk_runs.attempts +=
      after.asynchronous_run_attempts - before.asynchronous_run_attempts;
  telemetry->asynchronous_sdk_runs.successes +=
      after.asynchronous_run_successes - before.asynchronous_run_successes;
}

} // namespace

bool DecodedBackgroundRgbView::Valid() const noexcept {
  if (!data || width == 0 || height == 0 || asset_generation == 0)
    return false;
  const uint64_t row = static_cast<uint64_t>(width) * 3u;
  return row <= std::numeric_limits<std::size_t>::max() &&
         stride_bytes >= static_cast<std::size_t>(row);
}

class ProductionResidentFrameExecutor::Impl {
public:
  Impl(ProductionResidentRuntime runtime, IResidentDeviceOps *injected_ops)
      : runtime_(runtime),
        device_ops_(injected_ops ? injected_ops : &default_device_ops_),
        using_injected_device_ops_(injected_ops != nullptr) {}

  ~Impl() { DestroyAll(); }

  bool Configure(const ProductionResidentSetup &setup, std::string *error) {
    ++telemetry_.setup_attempts;
    if (configured_ && SameEffectiveSetup(setup_, setup)) {
      ++telemetry_.setup_reuses;
      if (error)
        error->clear();
      return true;
    }

    const bool preserve_runtime =
        configured_ && SameResourceIdentity(setup_, setup);
    configured_ = false;
    prepared_ = false;
    host_output_valid_ = false;
    pending_host_output_ = false;
    if (!ValidateSetup(setup, error))
      return false;
    if (!preserve_runtime) {
      DestroyAll();
      setup_ = setup;
      if (!PrepareRuntime(error))
        return false;
    } else {
      setup_ = setup;
      if (!device_ops_->Initialize(runtime_.cuda, setup_.runtime_identity,
                                   RequiredDeviceOps(), error) ||
          !PrepareOutputGeometry(error))
        return false;
    }

    if (!ConfigureEffects(error) || !PrepareBackground(error))
      return false;

    configured_ = true;
    prepared_ = false;
    last_error_.clear();
    ++telemetry_.setup_successes;
    if (error)
      error->clear();
    return true;
  }

  ResidentFrameKey Key() const noexcept {
    if (!configured_ || !stream_)
      return {};
    return ResidentFrameKey{setup_.configuration_generation, setup_.width,
                            setup_.height, setup_.runtime_identity,
                            reinterpret_cast<uintptr_t>(stream_)};
  }

  ResidentBoundaryResult Prepare(const ResidentFrameKey &key) noexcept {
    ++telemetry_.prepare_attempts;
    if (!configured_ || key != Key())
      return FailBoundary(
          "Resident executor key does not match prepared setup.",
          ResidentBoundaryResult::incompatible_output);
    prepared_ = true;
    host_output_valid_ = false;
    ++telemetry_.prepare_successes;
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult StageRgbToBgr(const HostRgbFrameView &host,
                                       const ResidentFrameKey &key) noexcept {
    auto &counter = telemetry_.cpu_stages[CpuStageIndex(
        ProductionCpuStageKind::rgb_to_bgr_staging)];
    ++counter.attempts;
    if (!Ready(key) || !host.ValidFor(key))
      return FailBoundary("Invalid RGB staging input.",
                          ResidentBoundaryResult::incompatible_output);
    video::Rgb24ToBgr24(host.data, host_bgr_input_.data(),
                        static_cast<int>(key.width),
                        static_cast<int>(key.height), host.stride_bytes,
                        static_cast<std::size_t>(key.width) * 3u);
    ++counter.successes;
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult Upload(const ResidentFrameKey &key,
                                ResidentImage &output) noexcept {
    output = {};
    if (!Ready(key))
      return ResidentBoundaryResult::incompatible_output;
    if (!Transfer(&cpu_bgr_input_, &canonical_[0], 1.0f,
                  ProductionTransferKind::host_upload))
      return ResidentBoundaryResult::runtime_failure;
    next_canonical_ = 1;
    matte_consumptions_ = 0;
    last_matte_ = {};
    tracked_vignette_center_valid_ = false;
    output = MakeResident(&canonical_[0], key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunMatte(const ResidentImage &current,
                                  uint64_t capture_sequence,
                                  uint64_t fingerprint,
                                  ResidentMatte &output) noexcept {
    output = {};
    if (!current.ValidFor(Key()) || !greenscreen_)
      return ResidentBoundaryResult::incompatible_output;
    ++telemetry_.matte_inferences.attempts;
    video::GpuFrame frame = Frame(current.image, nullptr, nullptr);
    const auto before = greenscreen_->execution_telemetry();
    const auto status = greenscreen_->Process(frame, nullptr);
    AccumulateExecutionDelta(before, greenscreen_->execution_telemetry(),
                             &telemetry_);
    if (status != NVCV_SUCCESS) {
      greenscreen_->InvalidateBindings();
      return FailBoundary("Maxine Green Screen run failed.");
    }
    const NvCVImage *matte = greenscreen_->MatteGpu();
    if (!matte)
      return FailBoundary("Maxine Green Screen produced no matte.",
                          ResidentBoundaryResult::incompatible_output);
    ++telemetry_.matte_inferences.successes;
    matte_consumptions_ = 0;
    output = ResidentMatte{
        matte,       NextIdentity(), current.image_identity, capture_sequence,
        fingerprint, current.key};
    last_matte_ = output;
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunStage(ResidentStageKind kind,
                                  const ResidentImage &current,
                                  const ResidentMatte *matte,
                                  ResidentImage &output) noexcept {
    output = {};
    if (kind >= ResidentStageKind::count || !StageEnabled(setup_, kind) ||
        !current.ValidFor(Key()))
      return ResidentBoundaryResult::incompatible_output;
    auto &stage_counter = telemetry_.stages[StageIndex(kind)];
    ++stage_counter.attempts;

    if (matte) {
      // Source identity/fingerprint/sequence are validated by
      // ResidentFrameSection before this call. Revalidate the concrete ABI
      // image and prepared key here at the production boundary.
      if (!ValidMatte(matte, current.key) ||
          matte->image != last_matte_.image ||
          matte->matte_identity != last_matte_.matte_identity ||
          matte->source_image_identity != last_matte_.source_image_identity ||
          matte->capture_sequence != last_matte_.capture_sequence ||
          matte->fingerprint != last_matte_.fingerprint ||
          matte->key != last_matte_.key) {
        return ResidentBoundaryResult::incompatible_output;
      }
      if (matte_consumptions_++ > 0)
        ++telemetry_.shared_matte_reuses;
    }

    ResidentBoundaryResult result = ResidentBoundaryResult::runtime_failure;
    switch (kind) {
    case ResidentStageKind::denoise:
      result = RunDenoise(current, output);
      break;
    case ResidentStageKind::eye_contact:
      result = RunEyeContact(current, output);
      break;
    case ResidentStageKind::background_blur:
      result = RunBackgroundBlur(current, matte, output);
      break;
    case ResidentStageKind::background_remove:
    case ResidentStageKind::background_replace:
      result = RunBackgroundComposite(kind, current, matte, output);
      break;
    case ResidentStageKind::relighting:
      result = RunRelight(current, matte, output);
      break;
    case ResidentStageKind::transfer:
      result = RunTransfer(current, output);
      break;
    case ResidentStageKind::auto_frame:
      result = RunAutoFrame(current, output);
      break;
    case ResidentStageKind::vignette:
      result = RunVignette(current, output);
      break;
    case ResidentStageKind::count:
      result = ResidentBoundaryResult::incompatible_output;
      break;
    }
    if (result == ResidentBoundaryResult::success)
      ++stage_counter.successes;
    return result;
  }

  ResidentBoundaryResult Download(ResidentReadbackBoundary boundary,
                                  const ResidentImage &current) noexcept {
    host_output_valid_ = false;
    if (!current.ValidFor(Key()))
      return ResidentBoundaryResult::incompatible_output;
    const NvCVImage *source = current.image;
    if (setup_.output_width != setup_.width ||
        setup_.output_height != setup_.height) {
      if (!device_ops_->Resize(*current.image, &resized_output_, stream_,
                               &last_error_))
        return ResidentBoundaryResult::runtime_failure;
      source = &resized_output_;
    }
    const auto kind = boundary == ResidentReadbackBoundary::final_output
                          ? ProductionTransferKind::final_download
                          : ProductionTransferKind::cpu_continuation_download;
    if (!Transfer(source, &cpu_bgr_output_, 1.0f, kind))
      return ResidentBoundaryResult::runtime_failure;
    pending_host_output_ = true;
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult Synchronize(const ResidentFrameKey &key) noexcept {
    if (!Ready(key) || !pending_host_output_)
      return ResidentBoundaryResult::incompatible_output;
    ++telemetry_.explicit_synchronizations.attempts;
    if (!device_ops_->Synchronize(stream_, &last_error_))
      return ResidentBoundaryResult::runtime_failure;
    ++telemetry_.explicit_synchronizations.successes;
    pending_host_output_ = false;
    host_output_valid_ = true;
    return ResidentBoundaryResult::success;
  }

  void InvalidateBindings() noexcept {
    prepared_ = false;
    host_output_valid_ = false;
    pending_host_output_ = false;
    if (greenscreen_)
      greenscreen_->InvalidateBindings();
    if (blur_)
      blur_->InvalidateBindings();
    if (relight_)
      relight_->InvalidateBindings();
    if (denoise_)
      denoise_->InvalidateBindings();
    if (transfer_)
      transfer_->InvalidateBindings();
    if (eye_contact_)
      eye_contact_->InvalidateBindings();
    if (auto_frame_)
      auto_frame_->InvalidateBindings();
  }

  HostBgrOutputView HostOutput() const noexcept {
    if (!host_output_valid_)
      return {};
    return HostBgrOutputView{
        host_bgr_output_.data(), setup_.output_width, setup_.output_height,
        static_cast<std::size_t>(setup_.output_width) * 3u};
  }

  const ProductionResidentTelemetry &telemetry() const noexcept {
    return telemetry_;
  }

  void ResetTelemetry() noexcept { telemetry_ = {}; }

  std::string_view last_error() const noexcept { return last_error_; }

private:
  enum class StreamOwner : uint8_t { none, vfx, ar, cuda };

  bool ValidateSetup(const ProductionResidentSetup &setup, std::string *error) {
    std::size_t ignored = 0;
    constexpr uint32_t valid_stage_mask =
        (uint32_t{1} << static_cast<uint32_t>(ResidentStageKind::count)) - 1u;
    if (setup.configuration_generation == 0 || setup.runtime_identity == 0 ||
        setup.enabled_maxine_stage_mask == 0 ||
        (setup.enabled_maxine_stage_mask & ~valid_stage_mask) != 0 ||
        !CheckedImageBytes(setup.width, setup.height, &ignored)) {
      return SetSetupError("Invalid resident setup identity or geometry.",
                           error);
    }
    if (setup.output_width == 0 || setup.output_height == 0 ||
        !CheckedImageBytes(setup.output_width, setup.output_height, &ignored)) {
      return SetSetupError("Invalid resident output geometry.", error);
    }
    if (!runtime_.nvcv || !runtime_.nvcv->IsInitialized() ||
        !runtime_.nvcv->f().NvCVImage_Init ||
        !runtime_.nvcv->f().NvCVImage_Alloc ||
        !runtime_.nvcv->f().NvCVImage_Dealloc ||
        !runtime_.nvcv->f().NvCVImage_Transfer) {
      return SetSetupError("Required NvCVImage runtime is unavailable.", error);
    }
    const bool needs_composite =
        StageEnabled(setup, ResidentStageKind::background_remove) ||
        StageEnabled(setup, ResidentStageKind::background_replace) ||
        StageEnabled(setup, ResidentStageKind::relighting);
    if (needs_composite && !runtime_.nvcv->f().NvCVImage_Composite)
      return SetSetupError("Required NvCVImage composite API is unavailable.",
                           error);
    if (EffectsNeedVfx(setup) &&
        (!runtime_.vfx || !runtime_.vfx->IsInitialized()))
      return SetSetupError("Required NvVFX runtime is unavailable.", error);
    if (EffectsNeedAr(setup) && (!runtime_.ar || !runtime_.ar->IsInitialized()))
      return SetSetupError("Required NvAR runtime is unavailable.", error);
    if ((!runtime_.cuda || !runtime_.cuda->IsInitialized()) &&
        !using_injected_device_ops_)
      return SetSetupError("Required CUDA driver runtime is unavailable.",
                           error);
    if (StageEnabled(setup, ResidentStageKind::background_replace) &&
        setup.effects.virtual_background.mode ==
            VirtualBackgroundMode::replace &&
        !setup.replacement_background.Valid()) {
      return SetSetupError(
          "Replacement mode requires resolved decoded background data.", error);
    }
    const uint32_t background_mask =
        setup.enabled_maxine_stage_mask &
        (ProductionResidentStageBit(ResidentStageKind::background_blur) |
         ProductionResidentStageBit(ResidentStageKind::background_remove) |
         ProductionResidentStageBit(ResidentStageKind::background_replace));
    const bool background_matches =
        (background_mask == 0) ||
        (background_mask ==
             ProductionResidentStageBit(ResidentStageKind::background_blur) &&
         setup.effects.virtual_background.mode ==
             VirtualBackgroundMode::blur) ||
        (background_mask ==
             ProductionResidentStageBit(ResidentStageKind::background_remove) &&
         setup.effects.virtual_background.mode ==
             VirtualBackgroundMode::remove) ||
        (background_mask == ProductionResidentStageBit(
                                ResidentStageKind::background_replace) &&
         setup.effects.virtual_background.mode ==
             VirtualBackgroundMode::replace);
    if (!background_matches)
      return SetSetupError(
          "Resident background stage mask does not match effective mode.",
          error);
    if ((StageEnabled(setup, ResidentStageKind::denoise) &&
         !setup.effects.video_noise_removal.enabled) ||
        (StageEnabled(setup, ResidentStageKind::eye_contact) &&
         !setup.effects.eye_contact.enabled) ||
        (StageEnabled(setup, ResidentStageKind::relighting) &&
         !setup.effects.virtual_key_light.enabled) ||
        (StageEnabled(setup, ResidentStageKind::auto_frame) &&
         !setup.effects.auto_frame.enabled) ||
        (StageEnabled(setup, ResidentStageKind::vignette) &&
         !setup.effects.vignette.enabled)) {
      return SetSetupError(
          "Resident stage mask enables a disabled effective effect.", error);
    }
    return true;
  }

  bool PrepareRuntime(std::string *error) {
    if (!CreateSharedStream(error) ||
        !device_ops_->Initialize(runtime_.cuda, setup_.runtime_identity,
                                 RequiredDeviceOps(), error) ||
        !PrepareFrameStorage(error) || !PrepareOutputGeometry(error)) {
      DestroyAll();
      return false;
    }
    return true;
  }

  uint32_t RequiredDeviceOps() const noexcept {
    uint32_t required_ops = 0;
    if (StageEnabled(setup_, ResidentStageKind::auto_frame))
      required_ops |= ResidentDeviceOpBit(ResidentDeviceOp::crop_scale);
    if (StageEnabled(setup_, ResidentStageKind::vignette))
      required_ops |= ResidentDeviceOpBit(ResidentDeviceOp::vignette);
    if (setup_.output_width != setup_.width ||
        setup_.output_height != setup_.height ||
        (StageEnabled(setup_, ResidentStageKind::background_replace) &&
         (setup_.replacement_background.width != setup_.width ||
          setup_.replacement_background.height != setup_.height)))
      required_ops |= ResidentDeviceOpBit(ResidentDeviceOp::resize);
    return required_ops;
  }

  bool CreateSharedStream(std::string *error) {
    if (runtime_.vfx && runtime_.vfx->IsInitialized() &&
        runtime_.vfx->f().NvVFX_CudaStreamCreate) {
      const auto status = runtime_.vfx->f().NvVFX_CudaStreamCreate(&stream_);
      if (status == NVCV_SUCCESS && stream_) {
        stream_owner_ = StreamOwner::vfx;
        return true;
      }
    }
    if (runtime_.ar && runtime_.ar->IsInitialized() &&
        runtime_.ar->f().NvAR_CudaStreamCreate) {
      const auto status = runtime_.ar->f().NvAR_CudaStreamCreate(&stream_);
      if (status == NVCV_SUCCESS && stream_) {
        stream_owner_ = StreamOwner::ar;
        return true;
      }
    }
    if (runtime_.cuda && runtime_.cuda->IsInitialized() &&
        runtime_.cuda->CreateStream(&stream_, error)) {
      stream_owner_ = StreamOwner::cuda;
      return true;
    }
    return SetSetupError("Unable to create a shared Maxine CUDA stream.",
                         error);
  }

  bool PrepareFrameStorage(std::string *error) {
    std::size_t bytes = 0;
    if (!CheckedImageBytes(setup_.width, setup_.height, &bytes))
      return SetSetupError("Resident frame geometry overflows storage.", error);
    host_bgr_input_.assign(bytes, 0);
    if (!InitCpuImage(&cpu_bgr_input_, setup_.width, setup_.height,
                      host_bgr_input_.data(), error) ||
        !AllocateImage(&canonical_[0], setup_.width, setup_.height, NVCV_BGR,
                       NVCV_U8, NVCV_CHUNKY, &canonical_allocated_[0], error) ||
        !AllocateImage(&canonical_[1], setup_.width, setup_.height, NVCV_BGR,
                       NVCV_U8, NVCV_CHUNKY, &canonical_allocated_[1], error))
      return false;
    return true;
  }

  bool PrepareOutputGeometry(std::string *error) {
    Deallocate(&resized_output_, &resized_output_allocated_);
    cpu_bgr_output_ = {};
    std::size_t bytes = 0;
    if (!CheckedImageBytes(setup_.output_width, setup_.output_height, &bytes))
      return SetSetupError("Resident output geometry overflows storage.",
                           error);
    host_bgr_output_.assign(bytes, 0);
    if (!InitCpuImage(&cpu_bgr_output_, setup_.output_width,
                      setup_.output_height, host_bgr_output_.data(), error))
      return false;
    if (setup_.output_width != setup_.width ||
        setup_.output_height != setup_.height) {
      if (!AllocateImage(&resized_output_, setup_.output_width,
                         setup_.output_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY,
                         &resized_output_allocated_, error))
        return false;
    }
    return true;
  }

  bool ConfigureEffects(std::string *error) {
    const bool needs_matte =
        StageEnabled(setup_, ResidentStageKind::background_blur) ||
        StageEnabled(setup_, ResidentStageKind::background_remove) ||
        StageEnabled(setup_, ResidentStageKind::background_replace) ||
        StageEnabled(setup_, ResidentStageKind::relighting);
    if (needs_matte && !greenscreen_) {
      greenscreen_ = std::make_unique<effects::VfxGreenScreenEffect>(
          runtime_.vfx, runtime_.nvcv, setup_.vfx_model_directory);
    }
    if (StageEnabled(setup_, ResidentStageKind::background_blur) && !blur_) {
      blur_ = std::make_unique<effects::VfxBackgroundBlurEffect>(
          runtime_.vfx, runtime_.nvcv, setup_.vfx_model_directory);
    }
    if (StageEnabled(setup_, ResidentStageKind::relighting) && !relight_) {
      relight_ = std::make_unique<effects::VfxRelightingEffect>(
          runtime_.vfx, runtime_.nvcv, setup_.vfx_model_directory);
    }
    if (StageEnabled(setup_, ResidentStageKind::denoise) && !denoise_) {
      denoise_ = std::make_unique<effects::VfxDenoiseEffect>(
          runtime_.vfx, runtime_.nvcv, setup_.vfx_model_directory);
    }
    if (StageEnabled(setup_, ResidentStageKind::transfer) && !transfer_) {
      transfer_ = std::make_unique<effects::VfxTransferEffect>(
          runtime_.vfx, runtime_.nvcv, setup_.vfx_model_directory,
          effects::VfxTransferEffect::OutputFormat{});
    }
    if (StageEnabled(setup_, ResidentStageKind::eye_contact) && !eye_contact_)
      eye_contact_ = std::make_unique<effects::ArEyeContactEffect>(runtime_.ar);
    if (StageEnabled(setup_, ResidentStageKind::auto_frame) && !auto_frame_)
      auto_frame_ = std::make_unique<effects::ArAutoFrameTracker>(runtime_.ar);

    if (!ConfigureVfx(greenscreen_.get(), error) ||
        !ConfigureVfx(blur_.get(), error) ||
        !ConfigureVfx(relight_.get(), error) ||
        !ConfigureVfx(denoise_.get(), error) ||
        !ConfigureVfx(transfer_.get(), error) || !ConfigureAr(error) ||
        !PrepareDenoiseBridge(error) || !PrepareRelightScratch(error))
      return false;
    return true;
  }

  template <typename Effect>
  bool ConfigureVfx(Effect *effect, std::string *error) {
    if (!effect)
      return true;
    return effect->SetExternalCudaStream(stream_, error) &&
           effect->Configure(setup_.effects, error) &&
           effect->Initialize(error);
  }

  bool ConfigureAr(std::string *error) {
    if (eye_contact_ && (!eye_contact_->SetExternalCudaStream(stream_, error) ||
                         !eye_contact_->Configure(setup_.effects, error)))
      return false;
    if (auto_frame_) {
      if (!auto_frame_->SetExternalCudaStream(stream_, error))
        return false;
      auto_frame_->SetOutputAspect(static_cast<float>(setup_.width) /
                                   static_cast<float>(setup_.height));
      effects::AutoFrameKnobs knobs{};
      knobs.strength = setup_.effects.auto_frame.strength;
      knobs.smoothing = setup_.effects.auto_frame.smoothing;
      knobs.headroom = setup_.effects.auto_frame.headroom;
      auto_frame_->SetKnobs(knobs);
      if (!auto_frame_->EnsureInitialized(&canonical_[0], error))
        return false;
    }
    return true;
  }

  bool PrepareDenoiseBridge(std::string *error) {
    if (!denoise_)
      return true;
    if (denoise_bridge_allocated_)
      return true;
    return AllocateImage(&denoise_bridge_, setup_.width, setup_.height,
                         NVCV_BGR, NVCV_F32, NVCV_PLANAR,
                         &denoise_bridge_allocated_, error);
  }

  bool PrepareRelightScratch(std::string *error) {
    if (!relight_)
      return true;
    if (relight_matte_allocated_)
      return true;
    return AllocateImage(&relight_matte_, setup_.width, setup_.height, NVCV_A,
                         NVCV_U8, NVCV_CHUNKY, &relight_matte_allocated_,
                         error);
  }

  bool PrepareBackground(std::string *error) {
    const auto mode = setup_.effects.virtual_background.mode;
    if (!StageEnabled(setup_, ResidentStageKind::background_remove) &&
        !StageEnabled(setup_, ResidentStageKind::background_replace))
      return true;
    const uint64_t desired_generation =
        mode == VirtualBackgroundMode::replace
            ? setup_.replacement_background.asset_generation
            : (uint64_t{1} << 63) ^
                  static_cast<uint64_t>(std::hash<std::string>{}(
                      setup_.effects.virtual_background.remove_color));
    if (background_allocated_ && background_generation_ == desired_generation)
      return true;

    Deallocate(&background_source_, &background_source_allocated_);
    Deallocate(&background_, &background_allocated_);
    background_cpu_ = {};
    background_host_.clear();

    uint32_t source_width = setup_.width;
    uint32_t source_height = setup_.height;
    std::size_t source_stride = static_cast<std::size_t>(source_width) * 3u;
    std::size_t bytes = 0;
    if (mode == VirtualBackgroundMode::replace) {
      source_width = setup_.replacement_background.width;
      source_height = setup_.replacement_background.height;
      source_stride = static_cast<std::size_t>(source_width) * 3u;
      if (!CheckedImageBytes(source_width, source_height, &bytes))
        return SetSetupError("Replacement background dimensions overflow.",
                             error);
      background_host_.assign(bytes, 0);
      video::Rgb24ToBgr24(
          setup_.replacement_background.data, background_host_.data(),
          static_cast<int>(source_width), static_cast<int>(source_height),
          setup_.replacement_background.stride_bytes, source_stride);
    } else {
      if (!CheckedImageBytes(source_width, source_height, &bytes))
        return SetSetupError("Solid background dimensions overflow.", error);
      std::array<uint8_t, 3> color{};
      if (!ParseRgbColor(setup_.effects.virtual_background.remove_color,
                         &color))
        return SetSetupError("Invalid background remove color.", error);
      background_host_.resize(bytes);
      for (std::size_t i = 0; i < bytes; i += 3u) {
        background_host_[i] = color[0];
        background_host_[i + 1] = color[1];
        background_host_[i + 2] = color[2];
      }
    }
    if (!InitCpuImage(&background_cpu_, source_width, source_height,
                      background_host_.data(), error))
      return false;

    NvCVImage *upload_target = &background_;
    bool *upload_allocated = &background_allocated_;
    const bool resize =
        source_width != setup_.width || source_height != setup_.height;
    if (resize) {
      upload_target = &background_source_;
      upload_allocated = &background_source_allocated_;
    }
    if (!AllocateImage(upload_target, source_width, source_height, NVCV_BGR,
                       NVCV_U8, NVCV_CHUNKY, upload_allocated, error) ||
        !Transfer(&background_cpu_, upload_target, 1.0f,
                  ProductionTransferKind::background_asset_setup_upload))
      return false;
    if (resize) {
      if (!AllocateImage(&background_, setup_.width, setup_.height, NVCV_BGR,
                         NVCV_U8, NVCV_CHUNKY, &background_allocated_, error) ||
          !device_ops_->Resize(background_source_, &background_, stream_,
                               error))
        return false;
    }
    background_generation_ = desired_generation;
    return true;
  }

  ResidentBoundaryResult RunDenoise(const ResidentImage &current,
                                    ResidentImage &output) noexcept {
    if (!denoise_ || !denoise_bridge_allocated_)
      return ResidentBoundaryResult::incompatible_output;
    if (!Transfer(current.image, &denoise_bridge_, 1.0f,
                  ProductionTransferKind::device_format_bridge))
      return ResidentBoundaryResult::runtime_failure;
    video::GpuFrame frame = Frame(&denoise_bridge_, nullptr, nullptr);
    const auto before = denoise_->execution_telemetry();
    const auto status = denoise_->Process(frame, nullptr);
    AccumulateExecutionDelta(before, denoise_->execution_telemetry(),
                             &telemetry_);
    if (status != NVCV_SUCCESS)
      return FailBoundary("Maxine denoise run failed.");
    NvCVImage *denoised = denoise_->OutputGpu();
    NvCVImage *destination = NextCanonical(current.image);
    if (!denoised || !Transfer(denoised, destination, 1.0f,
                               ProductionTransferKind::device_format_bridge))
      return ResidentBoundaryResult::runtime_failure;
    output = MakeResident(destination, current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunEyeContact(const ResidentImage &current,
                                       ResidentImage &output) noexcept {
    if (!eye_contact_)
      return ResidentBoundaryResult::incompatible_output;
    NvCVImage *destination = NextCanonical(current.image);
    video::GpuFrame frame = Frame(current.image, nullptr, destination);
    const auto before = eye_contact_->execution_telemetry();
    const auto status = eye_contact_->Process(frame, nullptr);
    AccumulateExecutionDelta(before, eye_contact_->execution_telemetry(),
                             &telemetry_);
    if (status != NVCV_SUCCESS)
      return FailBoundary("Maxine Eye Contact run failed.");
    output = MakeResident(destination, current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunBackgroundBlur(const ResidentImage &current,
                                           const ResidentMatte *matte,
                                           ResidentImage &output) noexcept {
    if (!blur_ || !ValidMatte(matte, current.key))
      return ResidentBoundaryResult::incompatible_output;
    video::GpuFrame frame = Frame(current.image, matte->image, nullptr);
    const auto before = blur_->execution_telemetry();
    const auto status = blur_->Process(frame, nullptr);
    AccumulateExecutionDelta(before, blur_->execution_telemetry(), &telemetry_);
    if (status != NVCV_SUCCESS)
      return FailBoundary("Maxine Background Blur run failed.");
    const NvCVImage *blurred = blur_->OutputGpu();
    if (!blurred)
      return ResidentBoundaryResult::incompatible_output;
    output = MakeResident(const_cast<NvCVImage *>(blurred), current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult
  RunBackgroundComposite(ResidentStageKind kind, const ResidentImage &current,
                         const ResidentMatte *matte,
                         ResidentImage &output) noexcept {
    const auto configured_mode = setup_.effects.virtual_background.mode;
    const bool matching = (kind == ResidentStageKind::background_remove &&
                           configured_mode == VirtualBackgroundMode::remove) ||
                          (kind == ResidentStageKind::background_replace &&
                           configured_mode == VirtualBackgroundMode::replace);
    if (!matching || !background_allocated_ || !ValidMatte(matte, current.key))
      return ResidentBoundaryResult::incompatible_output;
    NvCVImage *destination = NextCanonical(current.image);
    if (!Composite(current.image, &background_, matte->image, destination))
      return ResidentBoundaryResult::runtime_failure;
    output = MakeResident(destination, current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunRelight(const ResidentImage &current,
                                    const ResidentMatte *matte,
                                    ResidentImage &output) noexcept {
    if (!relight_ || !ValidMatte(matte, current.key))
      return ResidentBoundaryResult::incompatible_output;
    video::GpuFrame frame = Frame(current.image, matte->image, nullptr);
    const auto before = relight_->execution_telemetry();
    const auto status = relight_->Process(frame, nullptr);
    AccumulateExecutionDelta(before, relight_->execution_telemetry(),
                             &telemetry_);
    if (status != NVCV_SUCCESS)
      return FailBoundary("Maxine relighting run failed.");
    const NvCVImage *foreground = relight_->OutputGpu();
    if (!foreground)
      return ResidentBoundaryResult::incompatible_output;

    const NvCVImage *composite_matte = matte->image;
    const float intensity =
        Percent01(setup_.effects.virtual_key_light.intensity);
    if (intensity < 0.9999f) {
      if (!Transfer(matte->image, &relight_matte_, intensity,
                    ProductionTransferKind::device_format_bridge))
        return ResidentBoundaryResult::runtime_failure;
      composite_matte = &relight_matte_;
    }
    NvCVImage *destination = NextCanonical(current.image);
    if (!Composite(foreground, current.image, composite_matte, destination))
      return ResidentBoundaryResult::runtime_failure;
    output = MakeResident(destination, current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunTransfer(const ResidentImage &current,
                                     ResidentImage &output) noexcept {
    if (!transfer_)
      return ResidentBoundaryResult::incompatible_output;
    video::GpuFrame frame = Frame(current.image, nullptr, nullptr);
    const auto before = transfer_->execution_telemetry();
    const auto status = transfer_->Process(frame, nullptr);
    AccumulateExecutionDelta(before, transfer_->execution_telemetry(),
                             &telemetry_);
    if (status != NVCV_SUCCESS)
      return FailBoundary("Maxine Transfer run failed.");
    NvCVImage *transferred = transfer_->OutputGpu();
    if (!transferred)
      return ResidentBoundaryResult::incompatible_output;
    output = MakeResident(transferred, current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunAutoFrame(const ResidentImage &current,
                                      ResidentImage &output) noexcept {
    if (!auto_frame_)
      return ResidentBoundaryResult::incompatible_output;
    auto &cpu = telemetry_.cpu_stages[CpuStageIndex(
        ProductionCpuStageKind::auto_frame_tracking)];
    ++cpu.attempts;
    const auto before = auto_frame_->execution_telemetry();
    if (!auto_frame_->EnsureInitialized(current.image, &last_error_) ||
        !auto_frame_->Update(static_cast<int>(setup_.width),
                             static_cast<int>(setup_.height), &last_error_)) {
      AccumulateExecutionDelta(before, auto_frame_->execution_telemetry(),
                               &telemetry_);
      return ResidentBoundaryResult::runtime_failure;
    }
    AccumulateExecutionDelta(before, auto_frame_->execution_telemetry(),
                             &telemetry_);
    ++cpu.successes;
    const auto crop = auto_frame_->SmoothedCropPx();
    tracked_vignette_center_valid_ = auto_frame_->last_had_detection();
    if (tracked_vignette_center_valid_) {
      tracked_vignette_center_x_px_ = crop.x + crop.w * 0.5f;
      tracked_vignette_center_y_px_ = crop.y + crop.h * 0.5f;
    }
    NvCVImage *destination = NextCanonical(current.image);
    if (!device_ops_->CropScale(*current.image, destination, crop.x, crop.y,
                                crop.w, crop.h, stream_, &last_error_))
      return ResidentBoundaryResult::runtime_failure;
    output = MakeResident(destination, current.key);
    return ResidentBoundaryResult::success;
  }

  ResidentBoundaryResult RunVignette(const ResidentImage &current,
                                     ResidentImage &output) noexcept {
    if (!setup_.effects.vignette.enabled)
      return ResidentBoundaryResult::incompatible_output;
    NvCVImage *destination = NextCanonical(current.image);
    if (!Transfer(current.image, destination, 1.0f,
                  ProductionTransferKind::device_format_bridge))
      return ResidentBoundaryResult::runtime_failure;
    const float intensity = Percent01(setup_.effects.vignette.intensity);
    const bool use_tracked = setup_.effects.vignette.center_on_tracked_face &&
                             tracked_vignette_center_valid_;
    const float center_x = use_tracked ? tracked_vignette_center_x_px_
                                       : setup_.vignette_center_x_px;
    const float center_y = use_tracked ? tracked_vignette_center_y_px_
                                       : setup_.vignette_center_y_px;
    if (!device_ops_->Vignette(destination, intensity, center_x, center_y,
                               stream_, &last_error_))
      return ResidentBoundaryResult::runtime_failure;
    output = MakeResident(destination, current.key);
    return ResidentBoundaryResult::success;
  }

  bool Transfer(const NvCVImage *source, NvCVImage *destination, float scale,
                ProductionTransferKind kind) noexcept {
    auto &counter = telemetry_.transfers[TransferIndex(kind)];
    ++counter.attempts;
    const auto status = runtime_.nvcv->f().NvCVImage_Transfer(
        source, destination, scale, stream_, nullptr);
    if (status != NVCV_SUCCESS) {
      last_error_ = "NvCVImage_Transfer failed.";
      return false;
    }
    ++counter.successes;
    return true;
  }

  bool Composite(const NvCVImage *foreground, const NvCVImage *background,
                 const NvCVImage *matte, NvCVImage *destination) noexcept {
    ++telemetry_.composites.attempts;
    const auto status = runtime_.nvcv->f().NvCVImage_Composite(
        foreground, background, matte, destination, stream_);
    if (status != NVCV_SUCCESS) {
      last_error_ = "NvCVImage_Composite failed.";
      return false;
    }
    ++telemetry_.composites.successes;
    return true;
  }

  bool InitCpuImage(NvCVImage *image, uint32_t width, uint32_t height,
                    uint8_t *pixels, std::string *error) {
    const std::size_t stride = static_cast<std::size_t>(width) * 3u;
    const auto status = runtime_.nvcv->f().NvCVImage_Init(
        image, width, height, static_cast<int>(stride), pixels, NVCV_BGR,
        NVCV_U8, NVCV_CHUNKY, NVCV_CPU);
    if (status != NVCV_SUCCESS)
      return SetSetupError("NvCVImage_Init failed.", error);
    image->bufferBytes = stride * static_cast<std::size_t>(height);
    return true;
  }

  bool AllocateImage(NvCVImage *image, uint32_t width, uint32_t height,
                     NvCVImage_PixelFormat format,
                     NvCVImage_ComponentType component_type, unsigned layout,
                     bool *allocated, std::string *error) {
    if (!image || !allocated)
      return false;
    const auto status = runtime_.nvcv->f().NvCVImage_Alloc(
        image, width, height, format, component_type, layout, NVCV_GPU, 1);
    if (status != NVCV_SUCCESS)
      return SetSetupError("NvCVImage_Alloc failed.", error);
    *allocated = true;
    return true;
  }

  void Deallocate(NvCVImage *image, bool *allocated) noexcept {
    if (image && allocated && *allocated && runtime_.nvcv &&
        runtime_.nvcv->IsInitialized() && runtime_.nvcv->f().NvCVImage_Dealloc)
      (void)runtime_.nvcv->f().NvCVImage_Dealloc(image);
    if (image)
      *image = {};
    if (allocated)
      *allocated = false;
  }

  void DestroyAll() noexcept {
    auto_frame_.reset();
    eye_contact_.reset();
    transfer_.reset();
    denoise_.reset();
    relight_.reset();
    blur_.reset();
    greenscreen_.reset();

    Deallocate(&background_source_, &background_source_allocated_);
    Deallocate(&background_, &background_allocated_);
    Deallocate(&relight_matte_, &relight_matte_allocated_);
    Deallocate(&denoise_bridge_, &denoise_bridge_allocated_);
    Deallocate(&resized_output_, &resized_output_allocated_);
    Deallocate(&canonical_[1], &canonical_allocated_[1]);
    Deallocate(&canonical_[0], &canonical_allocated_[0]);

    if (stream_) {
      if (stream_owner_ == StreamOwner::vfx && runtime_.vfx &&
          runtime_.vfx->IsInitialized() &&
          runtime_.vfx->f().NvVFX_CudaStreamDestroy) {
        runtime_.vfx->f().NvVFX_CudaStreamDestroy(stream_);
      } else if (stream_owner_ == StreamOwner::ar && runtime_.ar &&
                 runtime_.ar->IsInitialized() &&
                 runtime_.ar->f().NvAR_CudaStreamDestroy) {
        (void)runtime_.ar->f().NvAR_CudaStreamDestroy(stream_);
      } else if (stream_owner_ == StreamOwner::cuda && runtime_.cuda &&
                 runtime_.cuda->IsInitialized()) {
        std::string ignored;
        (void)runtime_.cuda->DestroyStream(stream_, &ignored);
      }
    }
    stream_ = nullptr;
    stream_owner_ = StreamOwner::none;
    host_bgr_input_.clear();
    host_bgr_output_.clear();
    background_host_.clear();
    cpu_bgr_input_ = {};
    cpu_bgr_output_ = {};
    background_cpu_ = {};
    background_generation_ = 0;
    configured_ = false;
    prepared_ = false;
    host_output_valid_ = false;
    pending_host_output_ = false;
  }

  NvCVImage *NextCanonical(const NvCVImage *current) noexcept {
    if (current == &canonical_[0])
      next_canonical_ = 1;
    else if (current == &canonical_[1])
      next_canonical_ = 0;
    NvCVImage *result = &canonical_[next_canonical_];
    next_canonical_ ^= 1u;
    return result;
  }

  ResidentImage MakeResident(NvCVImage *image,
                             const ResidentFrameKey &key) noexcept {
    return ResidentImage{image, NextIdentity(), key};
  }

  uint64_t NextIdentity() noexcept {
    ++next_identity_;
    if (next_identity_ == 0)
      ++next_identity_;
    return next_identity_;
  }

  video::GpuFrame Frame(NvCVImage *input, const NvCVImage *matte,
                        NvCVImage *output) const noexcept {
    video::GpuFrame frame{};
    frame.width = static_cast<int>(setup_.width);
    frame.height = static_cast<int>(setup_.height);
    frame.nvcv_gpu = input;
    frame.matte_gpu = matte;
    frame.nvcv_tmp = output;
    frame.cuda_stream = stream_;
    return frame;
  }

  bool ValidMatte(const ResidentMatte *matte,
                  const ResidentFrameKey &key) const noexcept {
    if (!matte || !matte->image || matte->key != key)
      return false;
    NvCVImageValidationSpec spec{};
    spec.pixel_format = NVCV_A;
    spec.pixel_bytes = 1;
    spec.num_components = 1;
    return ValidateNvCVImage(*matte->image, spec) ==
               NvCVImageValidationStatus::ok &&
           matte->image->width == key.width &&
           matte->image->height == key.height;
  }

  bool Ready(const ResidentFrameKey &key) const noexcept {
    return configured_ && prepared_ && key == Key();
  }

  ResidentBoundaryResult
  FailBoundary(const char *message,
               ResidentBoundaryResult result =
                   ResidentBoundaryResult::runtime_failure) noexcept {
    try {
      last_error_ = message;
    } catch (...) {
    }
    return result;
  }

  bool SetSetupError(const char *message, std::string *error) {
    last_error_ = message;
    if (error)
      *error = last_error_;
    return false;
  }

  ProductionResidentRuntime runtime_{};
  DefaultResidentDeviceOps default_device_ops_{};
  IResidentDeviceOps *device_ops_ = nullptr;
  bool using_injected_device_ops_ = false;
  ProductionResidentSetup setup_{};
  ProductionResidentTelemetry telemetry_{};
  std::string last_error_;

  CUstream stream_ = nullptr;
  StreamOwner stream_owner_ = StreamOwner::none;
  bool configured_ = false;
  bool prepared_ = false;
  bool pending_host_output_ = false;
  bool host_output_valid_ = false;

  std::vector<uint8_t> host_bgr_input_;
  std::vector<uint8_t> host_bgr_output_;
  NvCVImage cpu_bgr_input_{};
  NvCVImage cpu_bgr_output_{};
  std::array<NvCVImage, 2> canonical_{};
  std::array<bool, 2> canonical_allocated_{};
  std::size_t next_canonical_ = 0;
  uint64_t next_identity_ = 1;

  NvCVImage resized_output_{};
  bool resized_output_allocated_ = false;
  NvCVImage denoise_bridge_{};
  bool denoise_bridge_allocated_ = false;
  NvCVImage relight_matte_{};
  bool relight_matte_allocated_ = false;

  std::vector<uint8_t> background_host_;
  NvCVImage background_cpu_{};
  NvCVImage background_source_{};
  bool background_source_allocated_ = false;
  NvCVImage background_{};
  bool background_allocated_ = false;
  uint64_t background_generation_ = 0;

  uint64_t matte_consumptions_ = 0;
  ResidentMatte last_matte_{};
  bool tracked_vignette_center_valid_ = false;
  float tracked_vignette_center_x_px_ = 0.0f;
  float tracked_vignette_center_y_px_ = 0.0f;

  std::unique_ptr<effects::VfxGreenScreenEffect> greenscreen_;
  std::unique_ptr<effects::VfxBackgroundBlurEffect> blur_;
  std::unique_ptr<effects::VfxRelightingEffect> relight_;
  std::unique_ptr<effects::VfxDenoiseEffect> denoise_;
  std::unique_ptr<effects::VfxTransferEffect> transfer_;
  std::unique_ptr<effects::ArEyeContactEffect> eye_contact_;
  std::unique_ptr<effects::ArAutoFrameTracker> auto_frame_;
};

ProductionResidentFrameExecutor::ProductionResidentFrameExecutor(
    ProductionResidentRuntime runtime, IResidentDeviceOps *device_ops)
    : impl_(std::make_unique<Impl>(runtime, device_ops)) {}

ProductionResidentFrameExecutor::~ProductionResidentFrameExecutor() = default;

bool ProductionResidentFrameExecutor::Configure(
    const ProductionResidentSetup &setup, std::string *error) {
  try {
    return impl_->Configure(setup, error);
  } catch (const std::bad_alloc &) {
    if (error)
      *error = "Resident executor setup allocation failed.";
    return false;
  } catch (...) {
    if (error)
      *error = "Resident executor setup failed.";
    return false;
  }
}

void ProductionResidentFrameExecutor::InvalidateBindings() noexcept {
  impl_->InvalidateBindings();
}

ResidentFrameKey ProductionResidentFrameExecutor::key() const noexcept {
  return impl_->Key();
}

uintptr_t ProductionResidentFrameExecutor::runtime_identity() const noexcept {
  return impl_->Key().runtime_identity;
}

uintptr_t ProductionResidentFrameExecutor::stream_identity() const noexcept {
  return impl_->Key().stream_identity;
}

HostBgrOutputView
ProductionResidentFrameExecutor::host_output() const noexcept {
  return impl_->HostOutput();
}

std::string_view ProductionResidentFrameExecutor::last_error() const noexcept {
  return impl_->last_error();
}

const ProductionResidentTelemetry &
ProductionResidentFrameExecutor::telemetry() const noexcept {
  return impl_->telemetry();
}

void ProductionResidentFrameExecutor::ResetTelemetry() noexcept {
  impl_->ResetTelemetry();
}

ResidentBoundaryResult
ProductionResidentFrameExecutor::Prepare(const ResidentFrameKey &key) noexcept {
  return impl_->Prepare(key);
}

ResidentBoundaryResult ProductionResidentFrameExecutor::StageRgbToBgr(
    const HostRgbFrameView &host, const ResidentFrameKey &key) noexcept {
  return impl_->StageRgbToBgr(host, key);
}

ResidentBoundaryResult ProductionResidentFrameExecutor::UploadStagedBgr(
    const ResidentFrameKey &key, ResidentImage &output) noexcept {
  return impl_->Upload(key, output);
}

ResidentBoundaryResult ProductionResidentFrameExecutor::RunSharedMatte(
    const ResidentImage &current, uint64_t capture_sequence,
    uint64_t matte_fingerprint, ResidentMatte &output) noexcept {
  return impl_->RunMatte(current, capture_sequence, matte_fingerprint, output);
}

ResidentBoundaryResult ProductionResidentFrameExecutor::RunCompatibleStage(
    ResidentStageKind kind, const ResidentImage &current,
    const ResidentMatte *matte, ResidentImage &output) noexcept {
  return impl_->RunStage(kind, current, matte, output);
}

ResidentBoundaryResult ProductionResidentFrameExecutor::DownloadToHost(
    ResidentReadbackBoundary boundary, const ResidentImage &current) noexcept {
  return impl_->Download(boundary, current);
}

ResidentBoundaryResult ProductionResidentFrameExecutor::Synchronize(
    ResidentReadbackBoundary, const ResidentFrameKey &key) noexcept {
  return impl_->Synchronize(key);
}

std::string_view
ProductionTransferKindName(ProductionTransferKind kind) noexcept {
  switch (kind) {
  case ProductionTransferKind::host_upload:
    return "host_upload";
  case ProductionTransferKind::final_download:
    return "final_download";
  case ProductionTransferKind::cpu_continuation_download:
    return "cpu_continuation_download";
  case ProductionTransferKind::device_format_bridge:
    return "device_format_bridge";
  case ProductionTransferKind::background_asset_setup_upload:
    return "background_asset_setup_upload";
  case ProductionTransferKind::count:
    break;
  }
  return "invalid";
}

std::string_view
ProductionCpuStageKindName(ProductionCpuStageKind kind) noexcept {
  switch (kind) {
  case ProductionCpuStageKind::rgb_to_bgr_staging:
    return "rgb_to_bgr_staging";
  case ProductionCpuStageKind::auto_frame_tracking:
    return "auto_frame_tracking";
  case ProductionCpuStageKind::count:
    break;
  }
  return "invalid";
}

} // namespace studiocast::maxine
