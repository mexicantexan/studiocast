#include "core/cuda/cuda_tensor.h"

#include <limits>

namespace studiocast::cuda {

namespace {

bool CheckedMul(std::size_t a, std::size_t b, std::size_t *out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  *out = a * b;
  return true;
}

bool CurrentTensorSize(const CudaTensor &tensor, CudaTensorSize *size_out) {
  CudaTensorSize local;
  CudaTensorSize *size = size_out ? size_out : &local;
  return CheckedNchwF32Size(tensor.n, tensor.c, tensor.h, tensor.w, size,
                            nullptr) &&
         tensor.bytes == size->bytes;
}

void ResetSize(CudaTensorSize *size_out) {
  if (size_out)
    *size_out = {};
}

} // namespace

bool CheckedNchwF32Size(int n, int c, int h, int w,
                        CudaTensorSize *size_out,
                        std::string *error_out) {
  if (error_out)
    error_out->clear();
  ResetSize(size_out);

  if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    if (error_out)
      *error_out = "CudaTensor NCHW F32 shape: invalid shape.";
    return false;
  }

  std::size_t elements = 1;
  if (!CheckedMul(elements, static_cast<std::size_t>(n), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(c), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(h), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(w), &elements)) {
    if (error_out)
      *error_out = "CudaTensor NCHW F32 shape: element count overflow.";
    return false;
  }

  std::size_t bytes = 0;
  if (!CheckedMul(elements, sizeof(float), &bytes)) {
    if (error_out)
      *error_out = "CudaTensor NCHW F32 shape: byte count overflow.";
    return false;
  }

  if (size_out) {
    size_out->elements = elements;
    size_out->bytes = bytes;
  }
  return true;
}

CudaTensor::CudaTensor(CudaTensor &&other) noexcept { MoveFrom(other); }

void CudaTensor::MoveFrom(CudaTensor &other) noexcept {
  ptr = other.ptr;
  pitch = other.pitch;
  bytes = other.bytes;
  n = other.n;
  c = other.c;
  h = other.h;
  w = other.w;
  owns_memory = other.owns_memory;
  other.ResetMetadata();
}

void CudaTensor::ResetMetadata() noexcept {
  ptr = 0;
  pitch = 0;
  bytes = 0;
  n = 0;
  c = 0;
  h = 0;
  w = 0;
  owns_memory = false;
}

bool CudaTensor::Valid() const {
  CudaTensorSize size;
  return ptr != 0 && CurrentTensorSize(*this, &size) && size.bytes > 0 &&
         pitch >= size.bytes;
}

std::size_t CudaTensor::ElementCount() const {
  CudaTensorSize size;
  if (!CheckedNchwF32Size(n, c, h, w, &size, nullptr))
    return 0;
  return size.elements;
}

bool CudaTensor::AllocateNchwF32(studiocast::cuda::CudaDriverApi *cuda,
                                 int n_in, int c_in, int h_in, int w_in,
                                 std::string *error_out) {
  if (error_out)
    error_out->clear();

  CudaTensorSize size;
  if (!CheckedNchwF32Size(n_in, c_in, h_in, w_in, &size, error_out))
    return false;

  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out =
          "CudaTensor::AllocateNchwF32: CUDA driver API not initialized.";
    return false;
  }

  studiocast::cuda::CudaDriverApi::PitchAllocation alloc{};
  // Allocate as a single row (Height=1) so the active tensor bytes are
  // contiguous.
  const bool ok = cuda->AllocatePitch(size.bytes, 1, &alloc, error_out,
                                      /*element_size_bytes=*/4);
  if (!ok)
    return false;

  ptr = alloc.ptr;
  pitch = alloc.pitch;
  bytes = size.bytes;
  n = n_in;
  c = c_in;
  h = h_in;
  w = w_in;
  owns_memory = true;
  return true;
}

bool CudaTensor::Free(studiocast::cuda::CudaDriverApi *cuda,
                      std::string *error_out) {
  if (error_out)
    error_out->clear();

  if (ptr == 0) {
    pitch = 0;
    bytes = 0;
    n = 0;
    c = 0;
    h = 0;
    w = 0;
    owns_memory = false;
    return true;
  }
  if (!owns_memory) {
    if (error_out)
      *error_out = "CudaTensor::Free called for non-owning tensor.";
    return false;
  }
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out = "CudaTensor::Free: CUDA driver API not initialized.";
    return false;
  }

  const bool ok = cuda->Free(ptr, error_out);
  if (!ok)
    return false;

  ptr = 0;
  pitch = 0;
  bytes = 0;
  n = 0;
  c = 0;
  h = 0;
  w = 0;
  owns_memory = false;
  return true;
}

bool CudaTensor::ReallocIfNeededNchwF32(studiocast::cuda::CudaDriverApi *cuda,
                                        int n_in, int c_in, int h_in, int w_in,
                                        std::string *error_out) {
  if (error_out)
    error_out->clear();

  CudaTensorSize want;
  if (!CheckedNchwF32Size(n_in, c_in, h_in, w_in, &want, error_out))
    return false;

  if (ptr != 0 && n == n_in && c == c_in && h == h_in && w == w_in) {
    // Capacity may be larger than required.
    if (bytes == want.bytes && pitch >= want.bytes)
      return true;
  }

  if (ptr != 0) {
    if (!Free(cuda, error_out))
      return false;
  }
  return AllocateNchwF32(cuda, n_in, c_in, h_in, w_in, error_out);
}

bool CudaTensor::UploadFromCpuF32(studiocast::cuda::CudaDriverApi *cuda,
                                  const float *src, std::size_t src_floats,
                                  studiocast::cuda::CUstream stream,
                                  std::string *error_out) const {
  if (error_out)
    error_out->clear();

  CudaTensorSize size;
  if (!Valid() || !CurrentTensorSize(*this, &size)) {
    if (error_out)
      *error_out = "CudaTensor::UploadFromCpuF32: invalid tensor.";
    return false;
  }
  if (!src) {
    if (error_out)
      *error_out = "CudaTensor::UploadFromCpuF32: src is null.";
    return false;
  }
  if (src_floats < size.elements) {
    if (error_out)
      *error_out = "CudaTensor::UploadFromCpuF32: source buffer too small.";
    return false;
  }
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out =
          "CudaTensor::UploadFromCpuF32: CUDA driver API not initialized.";
    return false;
  }

  return cuda->MemcpyHtoD2DAsync(ptr, pitch, src, /*src_pitch=*/size.bytes,
                                 /*width_bytes=*/size.bytes, /*height=*/1,
                                 stream, error_out);
}

bool CudaTensor::DownloadToCpuF32(studiocast::cuda::CudaDriverApi *cuda,
                                  std::vector<float> *out,
                                  studiocast::cuda::CUstream stream,
                                  std::string *error_out) const {
  if (error_out)
    error_out->clear();
  if (!out)
    return false;
  out->clear();

  CudaTensorSize size;
  if (!Valid() || !CurrentTensorSize(*this, &size)) {
    if (error_out)
      *error_out = "CudaTensor::DownloadToCpuF32: invalid tensor.";
    return false;
  }
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out =
          "CudaTensor::DownloadToCpuF32: CUDA driver API not initialized.";
    return false;
  }

  out->resize(size.elements);
  return cuda->MemcpyDtoH2DAsync(out->data(), /*dst_pitch=*/size.bytes, ptr,
                                 pitch, /*width_bytes=*/size.bytes,
                                 /*height=*/1, stream, error_out);
}

} // namespace studiocast::cuda
