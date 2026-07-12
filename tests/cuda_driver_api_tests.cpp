#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/maxine/cuda_driver_api.h"

#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

bool Expect(bool cond, const std::string &msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
  }
  return cond;
}

bool ExpectEq(const std::string &name, int got, int want) {
  if (got != want) {
    std::cerr << "[FAIL] " << name << ": got " << got << ", want " << want
              << "\n";
    return false;
  }
  return true;
}

bool ExpectEqSize(const std::string &name, std::size_t got, std::size_t want) {
  if (got != want) {
    std::cerr << "[FAIL] " << name << ": got " << got << ", want " << want
              << "\n";
    return false;
  }
  return true;
}

bool Contains(const std::string &s, const std::string &needle) {
  return s.find(needle) != std::string::npos;
}

bool TestDeviceOrdinalConfiguration() {
  studiocast::maxine::CudaDriverApi cuda;
  std::string err;

  bool ok = true;
  ok = ExpectEq("default CUDA device ordinal", cuda.device_ordinal(), 0) && ok;
  ok = Expect(cuda.SetDeviceOrdinal(2, &err),
              "SetDeviceOrdinal should accept non-negative ordinals") &&
       ok;
  ok = Expect(err.empty(), "SetDeviceOrdinal should clear error on success") &&
       ok;
  ok = ExpectEq("configured CUDA device ordinal", cuda.device_ordinal(), 2) &&
       ok;

  ok = Expect(!cuda.SetDeviceOrdinal(-1, &err),
              "SetDeviceOrdinal should reject negative ordinals") &&
       ok;
  ok = Expect(Contains(err, "non-negative"),
              "negative ordinal error should explain the constraint") &&
       ok;
  ok = ExpectEq("failed SetDeviceOrdinal preserves ordinal",
                cuda.device_ordinal(), 2) &&
       ok;

  studiocast::maxine::CudaDriverApi moved(std::move(cuda));
  ok = ExpectEq("move constructor preserves CUDA device ordinal",
                moved.device_ordinal(), 2) &&
       ok;

  studiocast::maxine::CudaDriverApi assigned;
  ok = Expect(assigned.SetDeviceOrdinal(1, &err),
              "pre-assignment SetDeviceOrdinal should succeed") &&
       ok;
  assigned = std::move(moved);
  ok = ExpectEq("move assignment preserves CUDA device ordinal",
                assigned.device_ordinal(), 2) &&
       ok;

  return ok;
}

bool TestCudaBufferWrappersAreMoveOnly() {
  using studiocast::cuda::CudaImage;
  using studiocast::cuda::CudaTensor;
  using studiocast::cuda::PixelFormatGpu;

  static_assert(!std::is_copy_constructible_v<CudaImage>);
  static_assert(!std::is_copy_assignable_v<CudaImage>);
  static_assert(std::is_nothrow_move_constructible_v<CudaImage>);
  static_assert(!std::is_move_assignable_v<CudaImage>);

  static_assert(!std::is_copy_constructible_v<CudaTensor>);
  static_assert(!std::is_copy_assignable_v<CudaTensor>);
  static_assert(std::is_nothrow_move_constructible_v<CudaTensor>);
  static_assert(!std::is_move_assignable_v<CudaTensor>);

  bool ok = true;

  CudaImage image;
  image.ptr = 0x1234;
  image.pitch = 64;
  image.w = 16;
  image.h = 8;
  image.format = PixelFormatGpu::rgba_u8;
  image.owns_memory = true;

  CudaImage moved_image(std::move(image));
  ok = Expect(moved_image.ptr == 0x1234 && moved_image.pitch == 64 &&
                  moved_image.w == 16 && moved_image.h == 8 &&
                  moved_image.format == PixelFormatGpu::rgba_u8 &&
                  moved_image.owns_memory,
              "CudaImage move constructor should transfer metadata") &&
       ok;
  ok = Expect(image.ptr == 0 && image.pitch == 0 && image.w == 0 &&
                  image.h == 0 && !image.owns_memory,
              "CudaImage move constructor should clear source") &&
       ok;

  CudaImage non_owning_image;
  non_owning_image.ptr = 0x5678;
  non_owning_image.pitch = 32;
  non_owning_image.w = 4;
  non_owning_image.h = 4;
  non_owning_image.owns_memory = false;
  non_owning_image.ClearMetadata();
  ok = Expect(non_owning_image.ptr == 0 && non_owning_image.pitch == 0 &&
                  non_owning_image.w == 0 && non_owning_image.h == 0 &&
                  !non_owning_image.owns_memory,
              "CudaImage ClearMetadata should reset non-owning views") &&
       ok;

  CudaTensor tensor;
  tensor.ptr = 0x9abc;
  tensor.pitch = 256;
  tensor.bytes = 128;
  tensor.n = 1;
  tensor.c = 3;
  tensor.h = 4;
  tensor.w = 5;
  tensor.owns_memory = true;

  CudaTensor moved_tensor(std::move(tensor));
  ok = Expect(moved_tensor.ptr == 0x9abc && moved_tensor.pitch == 256 &&
                  moved_tensor.bytes == 128 && moved_tensor.n == 1 &&
                  moved_tensor.c == 3 && moved_tensor.h == 4 &&
                  moved_tensor.w == 5 && moved_tensor.owns_memory,
              "CudaTensor move constructor should transfer metadata") &&
       ok;
  ok = Expect(tensor.ptr == 0 && tensor.pitch == 0 && tensor.bytes == 0 &&
                  tensor.n == 0 && tensor.c == 0 && tensor.h == 0 &&
                  tensor.w == 0 && !tensor.owns_memory,
              "CudaTensor move constructor should clear source") &&
       ok;

  return ok;
}

bool TestCudaTensorCheckedShapeMathNoGpu() {
  using studiocast::cuda::CheckedNchwF32Size;
  using studiocast::cuda::CudaTensor;
  using studiocast::cuda::CudaTensorSize;

  bool ok = true;
  std::string err;
  CudaTensorSize size;

  ok = Expect(CheckedNchwF32Size(2, 3, 4, 5, &size, &err),
              "valid NCHW F32 shape should compute") &&
       ok;
  ok = Expect(err.empty(), "valid shape should clear error") && ok;
  ok = ExpectEqSize("valid NCHW element count", size.elements, 120u) && ok;
  ok = ExpectEqSize("valid NCHW byte count", size.bytes,
                    120u * sizeof(float)) &&
       ok;
  const CudaTensorSize valid_size = size;

  ok = Expect(!CheckedNchwF32Size(1, 3, 0, 5, &size, &err),
              "zero dimension should be rejected") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "zero dimension error should explain invalid shape") &&
       ok;
  ok = ExpectEqSize("failed shape clears element count", size.elements, 0u) &&
       ok;
  ok = ExpectEqSize("failed shape clears byte count", size.bytes, 0u) && ok;

  ok = Expect(!CheckedNchwF32Size(1, -3, 4, 5, &size, &err),
              "negative dimension should be rejected") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "negative dimension error should explain invalid shape") &&
       ok;

  const int max_int = std::numeric_limits<int>::max();
  ok = Expect(!CheckedNchwF32Size(max_int, max_int, max_int, max_int, &size,
                                  &err),
              "overflowing NCHW shape should be rejected") &&
       ok;
  ok = Expect(Contains(err, "overflow"),
              "overflowing shape error should mention overflow") &&
       ok;

  CudaTensor overflow;
  overflow.ptr = 0x5678;
  overflow.pitch = std::numeric_limits<std::size_t>::max();
  overflow.bytes = std::numeric_limits<std::size_t>::max();
  overflow.n = max_int;
  overflow.c = max_int;
  overflow.h = max_int;
  overflow.w = max_int;
  overflow.owns_memory = true;
  ok = Expect(!overflow.Valid(),
              "CudaTensor with overflowing shape should be invalid") &&
       ok;
  ok = ExpectEqSize("overflowing CudaTensor element count",
                    overflow.ElementCount(), 0u) &&
       ok;

  CudaTensor empty;
  ok = Expect(!empty.Valid(), "empty CudaTensor should be invalid") && ok;
  ok = ExpectEqSize("empty CudaTensor element count", empty.ElementCount(),
                    0u) &&
       ok;

  CudaTensor tensor;
  tensor.ptr = 0x9abc;
  tensor.pitch = valid_size.bytes;
  tensor.bytes = valid_size.bytes;
  tensor.n = 2;
  tensor.c = 3;
  tensor.h = 4;
  tensor.w = 5;
  tensor.owns_memory = true;
  ok = Expect(tensor.Valid(),
              "manually populated tensor with checked size should be valid") &&
       ok;
  ok = ExpectEqSize("valid CudaTensor element count", tensor.ElementCount(),
                    120u) &&
       ok;

  CudaTensor moved(std::move(tensor));
  ok = Expect(tensor.ptr == 0 && tensor.pitch == 0 && tensor.bytes == 0 &&
                  tensor.n == 0 && tensor.c == 0 && tensor.h == 0 &&
                  tensor.w == 0 && !tensor.owns_memory,
              "moved-from CudaTensor should reset metadata") &&
       ok;
  ok = Expect(moved.Valid(), "moved CudaTensor should preserve valid shape") &&
       ok;

  CudaTensor mismatch;
  mismatch.ptr = 0x1234;
  mismatch.pitch = 1024;
  mismatch.bytes = 128;
  mismatch.n = 1;
  mismatch.c = 3;
  mismatch.h = 4;
  mismatch.w = 5;
  mismatch.owns_memory = true;
  ok = Expect(!mismatch.Valid(),
              "CudaTensor with mismatched byte count should be invalid") &&
       ok;

  float value = 0.0f;
  ok = Expect(!mismatch.UploadFromCpuF32(nullptr, &value, 1, 0, &err),
              "upload validation should reject bad tensor before CUDA use") &&
       ok;
  ok = Expect(Contains(err, "invalid tensor"),
              "bad upload tensor should report invalid tensor") &&
       ok;

  std::vector<float> out(1, 1.0f);
  ok = Expect(!mismatch.DownloadToCpuF32(nullptr, &out, 0, &err),
              "download validation should reject bad tensor before CUDA use") &&
       ok;
  ok = Expect(Contains(err, "invalid tensor"),
              "bad download tensor should report invalid tensor") &&
       ok;
  ok = Expect(out.empty(),
              "failed download validation should leave output vector empty") &&
       ok;

  ok = Expect(!empty.AllocateNchwF32(nullptr, 1, 0, 4, 5, &err),
              "invalid allocation shape should fail without CUDA") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "invalid allocation shape should report shape error") &&
       ok;
  ok = Expect(!empty.AllocateNchwF32(nullptr, max_int, max_int, max_int,
                                     max_int, &err),
              "overflowing allocation shape should fail without CUDA") &&
       ok;
  ok = Expect(Contains(err, "overflow"),
              "overflowing allocation shape should report overflow") &&
       ok;

  const auto moved_ptr = moved.ptr;
  ok = Expect(!moved.ReallocIfNeededNchwF32(nullptr, 1, 0, 4, 5, &err),
              "invalid reallocation shape should fail without freeing") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "invalid reallocation shape should report shape error") &&
       ok;
  ok = Expect(moved.ptr == moved_ptr,
              "invalid reallocation shape should preserve existing tensor") &&
       ok;

  return ok;
}

} // namespace

int main() {
  if (!TestDeviceOrdinalConfiguration())
    return 1;
  if (!TestCudaBufferWrappersAreMoveOnly())
    return 1;
  if (!TestCudaTensorCheckedShapeMathNoGpu())
    return 1;
  std::cout << "CUDA DRIVER API TESTS OK\n";
  return 0;
}
