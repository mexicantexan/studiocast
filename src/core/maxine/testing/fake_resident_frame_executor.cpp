#include "core/maxine/testing/fake_resident_frame_executor.h"

#include <limits>

namespace studiocast::maxine::testing {
namespace {

void ConfigureImage(NvCVImage &image, const ResidentFrameKey &key,
                    NvCVImage_PixelFormat format, uint8_t pixel_bytes,
                    uint8_t components, uintptr_t fake_address) noexcept {
  image = {};
  image.width = key.width;
  image.height = key.height;
  image.pitch = static_cast<int32_t>(key.width * pixel_bytes);
  image.pixelFormat = format;
  image.componentType = NVCV_U8;
  image.pixelBytes = pixel_bytes;
  image.componentBytes = 1;
  image.numComponents = components;
  image.planar = static_cast<uint8_t>(NVCV_CHUNKY);
  image.gpuMem = static_cast<uint8_t>(NVCV_GPU);
  image.pixels = reinterpret_cast<void *>(fake_address);
  image.bufferBytes = static_cast<uint64_t>(image.pitch) * key.height;
}

} // namespace

void FakeResidentFrameExecutor::ConfigureImages(
    const ResidentFrameKey &key) noexcept {
  ConfigureImage(images_[0], key, NVCV_BGR, 3, 3, 0x1000);
  ConfigureImage(images_[1], key, NVCV_BGR, 3, 3, 0x2000);
  ConfigureImage(images_[2], key, NVCV_BGR, 3, 3, 0x3000);
  ConfigureImage(matte_image_, key, NVCV_A, 1, 1, 0x4000);
  next_image_slot_ = 0;
}

void FakeResidentFrameExecutor::DoSyntheticWork(uint32_t multiplier) noexcept {
  const uint64_t iterations =
      static_cast<uint64_t>(synthetic_work_iterations_) * multiplier;
  for (uint64_t i = 0; i < iterations; ++i)
    work_sink_ = (work_sink_ * 1664525u) + i + 1013904223u;
}

ResidentImage FakeResidentFrameExecutor::NextImage() noexcept {
  NvCVImage *image = &images_[next_image_slot_];
  next_image_slot_ = (next_image_slot_ + 1) % images_.size();
  return ResidentImage{image, next_identity_++, prepared_key_};
}

ResidentBoundaryResult
FakeResidentFrameExecutor::Prepare(const ResidentFrameKey &key) noexcept {
  ++calls_.prepare;
  DoSyntheticWork();
  if (prepare_result_ != ResidentBoundaryResult::success)
    return prepare_result_;
  prepared_key_ = key;
  ConfigureImages(key);
  return ResidentBoundaryResult::success;
}

ResidentBoundaryResult FakeResidentFrameExecutor::StageRgbToBgr(
    const HostRgbFrameView &, const ResidentFrameKey &) noexcept {
  ++calls_.rgb_to_bgr;
  DoSyntheticWork();
  return rgb_to_bgr_result_;
}

ResidentBoundaryResult FakeResidentFrameExecutor::UploadStagedBgr(
    const ResidentFrameKey &, ResidentImage &output) noexcept {
  ++calls_.upload;
  DoSyntheticWork(2);
  if (upload_result_ != ResidentBoundaryResult::success)
    return upload_result_;
  output = NextImage();
  calls_.uploaded_image_identity = output.image_identity;
  return ResidentBoundaryResult::success;
}

ResidentBoundaryResult FakeResidentFrameExecutor::RunSharedMatte(
    const ResidentImage &current, uint64_t capture_sequence,
    uint64_t matte_fingerprint, ResidentMatte &output) noexcept {
  ++calls_.matte;
  calls_.matte_source_identity = current.image_identity;
  calls_.last_matte_fingerprint = matte_fingerprint;
  DoSyntheticWork(3);
  if (matte_result_ != ResidentBoundaryResult::success)
    return matte_result_;
  output = ResidentMatte{&matte_image_, next_identity_++, current.image_identity,
                         capture_sequence, matte_fingerprint, current.key};
  if (corrupt_next_matte_output_) {
    output.source_image_identity += 1;
    corrupt_next_matte_output_ = false;
  }
  return ResidentBoundaryResult::success;
}

ResidentBoundaryResult FakeResidentFrameExecutor::RunCompatibleStage(
    ResidentStageKind kind, const ResidentImage &current,
    const ResidentMatte *, ResidentImage &output) noexcept {
  const std::size_t index = calls_.stage_count;
  if (index < calls_.stage_input_identities.size()) {
    calls_.stage_input_identities[index] = current.image_identity;
    calls_.stage_kinds[index] = kind;
  }
  ++calls_.stage_count;
  DoSyntheticWork(2);
  if (kind == failing_stage_)
    return stage_failure_result_;
  output = NextImage();
  if (corrupt_next_stage_output_) {
    output.key.stream_identity += 1;
    corrupt_next_stage_output_ = false;
  }
  if (index < calls_.stage_output_identities.size())
    calls_.stage_output_identities[index] = output.image_identity;
  return ResidentBoundaryResult::success;
}

ResidentBoundaryResult FakeResidentFrameExecutor::DownloadToHost(
    ResidentReadbackBoundary boundary, const ResidentImage &) noexcept {
  ++calls_.download;
  calls_.last_download_boundary = boundary;
  DoSyntheticWork(2);
  return download_result_;
}

ResidentBoundaryResult FakeResidentFrameExecutor::Synchronize(
    ResidentReadbackBoundary boundary, const ResidentFrameKey &) noexcept {
  ++calls_.synchronize;
  calls_.last_sync_boundary = boundary;
  DoSyntheticWork();
  return synchronize_result_;
}

void FakeResidentFrameExecutor::ClearFailures() noexcept {
  prepare_result_ = ResidentBoundaryResult::success;
  rgb_to_bgr_result_ = ResidentBoundaryResult::success;
  upload_result_ = ResidentBoundaryResult::success;
  matte_result_ = ResidentBoundaryResult::success;
  failing_stage_ = ResidentStageKind::count;
  stage_failure_result_ = ResidentBoundaryResult::runtime_failure;
  download_result_ = ResidentBoundaryResult::success;
  synchronize_result_ = ResidentBoundaryResult::success;
  corrupt_next_stage_output_ = false;
  corrupt_next_matte_output_ = false;
}

} // namespace studiocast::maxine::testing
