#include "core/maxine/nvcv_types.h"
#include "core/maxine/cuda_crop_scale.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace {

using studiocast::maxine::NVCV_BGR;
using studiocast::maxine::NVCV_CHUNKY;
using studiocast::maxine::NVCV_CPU;
using studiocast::maxine::NVCV_GPU;
using studiocast::maxine::NVCV_PLANAR;
using studiocast::maxine::NVCV_RGB;
using studiocast::maxine::NVCV_U16;
using studiocast::maxine::NVCV_U8;
using studiocast::maxine::NvCVImage;
using studiocast::maxine::NvCVImageValidationSpec;
using studiocast::maxine::NvCVImageValidationStatus;
using studiocast::maxine::ValidateBgrU8CudaNvCVImage;
using studiocast::maxine::ValidateNvCVImage;

void *FakePixels() {
  return reinterpret_cast<void *>(static_cast<std::uintptr_t>(0x1000));
}

NvCVImage ValidBgrU8CudaImage() {
  NvCVImage image{};
  image.width = 4;
  image.height = 3;
  image.pitch = 16;
  image.pixelFormat = NVCV_BGR;
  image.componentType = NVCV_U8;
  image.pixelBytes = 3;
  image.componentBytes = 1;
  image.numComponents = 3;
  image.planar = static_cast<std::uint8_t>(NVCV_CHUNKY);
  image.gpuMem = static_cast<std::uint8_t>(NVCV_GPU);
  image.pixels = FakePixels();
  image.bufferBytes = 48;
  return image;
}

bool ExpectStatus(const std::string &name, NvCVImageValidationStatus got,
                  NvCVImageValidationStatus want) {
  if (got != want) {
    std::cerr << "[FAIL] " << name << ": got "
              << static_cast<int>(got) << ", want " << static_cast<int>(want)
              << "\n";
    return false;
  }
  return true;
}

bool TestValidImagePasses() {
  return ExpectStatus("valid BGR/U8 CUDA image",
                      ValidateBgrU8CudaNvCVImage(ValidBgrU8CudaImage()),
                      NvCVImageValidationStatus::ok);
}

bool TestMetadataValidation() {
  bool ok = true;

  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.gpuMem = static_cast<std::uint8_t>(NVCV_CPU);
    ok = ExpectStatus("unexpected gpuMem",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_gpu_mem) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.pixelFormat = NVCV_RGB;
    ok = ExpectStatus("unexpected format",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_pixel_format) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.componentType = NVCV_U16;
    ok = ExpectStatus("unexpected component type",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_component_type) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.planar = static_cast<std::uint8_t>(NVCV_PLANAR);
    ok = ExpectStatus("unexpected layout",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_layout) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.componentBytes = 2;
    ok = ExpectStatus("unexpected component bytes",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_component_bytes) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.numComponents = 4;
    ok = ExpectStatus("unexpected components",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_num_components) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.pixelBytes = 4;
    ok = ExpectStatus("unexpected pixel bytes",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::unexpected_pixel_bytes) &&
         ok;
  }

  return ok;
}

bool TestExtentAndPointerValidation() {
  bool ok = true;

  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.width = 0;
    ok = ExpectStatus("zero dimensions rejected by default",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::zero_dimensions) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.width = 0;
    image.pixels = nullptr;
    image.pitch = 0;
    image.bufferBytes = 0;
    ok = ExpectStatus("zero dimensions can be accepted for no-op kernels",
                      ValidateBgrU8CudaNvCVImage(
                          image, /*allow_zero_dimensions=*/true),
                      NvCVImageValidationStatus::ok) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.pixels = nullptr;
    ok = ExpectStatus("null pixels rejected for non-empty images",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::null_pixels) &&
         ok;
  }

  return ok;
}

bool TestPitchAndBufferValidation() {
  bool ok = true;

  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.pitch = 0;
    ok = ExpectStatus("zero pitch rejected",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::invalid_pitch) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.pitch = -1;
    ok = ExpectStatus("negative pitch rejected",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::invalid_pitch) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.pitch = 11;
    ok = ExpectStatus("pitch smaller than row rejected",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::pitch_too_small) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.bufferBytes = 43;
    ok = ExpectStatus("buffer smaller than pitched extent rejected",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::buffer_too_small) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.bufferBytes = 44;
    ok = ExpectStatus("minimum pitched extent accepted",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::ok) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.bufferBytes = 0;
    ok = ExpectStatus("missing required buffer size rejected",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::buffer_too_small) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.bufferBytes = 0;
    NvCVImageValidationSpec spec{};
    spec.require_buffer_bytes = false;
    ok = ExpectStatus("unknown buffer size can be accepted by explicit policy",
                      ValidateNvCVImage(image, spec),
                      NvCVImageValidationStatus::ok) &&
         ok;
  }
  {
    NvCVImage image = ValidBgrU8CudaImage();
    image.width =
        static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max() / 3) +
        1u;
    image.pitch = std::numeric_limits<std::int32_t>::max();
    image.bufferBytes = std::numeric_limits<std::uint64_t>::max();
    ok = ExpectStatus("row bytes outside signed pitch range rejected",
                      ValidateBgrU8CudaNvCVImage(image),
                      NvCVImageValidationStatus::
                          row_bytes_exceed_pitch_range) &&
         ok;
  }

  return ok;
}

bool TestMaxineBgrResizeRoundingContract() {
  using studiocast::maxine::detail::RoundClampBgrResizeBilinearU8;

  struct RoundCase {
    float value;
    std::uint8_t expected;
    const char *name;
  };

  const RoundCase cases[] = {
      {-0.25f, 0, "negative clamps before rounding"},
      {0.49f, 0, "near tie below half rounds down"},
      {0.5f, 1, "half tie rounds up"},
      {1.49f, 1, "nonzero near tie below half rounds down"},
      {1.5f, 2, "nonzero half tie rounds up"},
      {42.75f, 43, "ordinary fractional value rounds up"},
      {254.49f, 254, "upper near tie below half rounds down"},
      {254.5f, 255, "upper half tie rounds up"},
      {300.0f, 255, "above byte max clamps"},
  };

  bool ok = true;
  for (const RoundCase &tc : cases) {
    const std::uint8_t actual = RoundClampBgrResizeBilinearU8(tc.value);
    if (actual != tc.expected) {
      std::cerr << "[FAIL] Maxine BGR resize rounding " << tc.name
                << ": got " << static_cast<int>(actual) << ", want "
                << static_cast<int>(tc.expected) << "\n";
      ok = false;
    }
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = TestValidImagePasses() && ok;
  ok = TestMetadataValidation() && ok;
  ok = TestExtentAndPointerValidation() && ok;
  ok = TestPitchAndBufferValidation() && ok;
  ok = TestMaxineBgrResizeRoundingContract() && ok;

  if (!ok)
    return 1;

  std::cout << "NVCV VALIDATION TESTS OK\n";
  return 0;
}
