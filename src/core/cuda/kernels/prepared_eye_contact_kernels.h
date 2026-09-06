#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"

#include <string>

namespace studiocast::cuda::kernels {

struct PreparedEyeContactKernels {
  CudaDriverApi *driver = nullptr;
  CUcontext owner_context = nullptr;
  CUstream owner_stream = nullptr;
  CUmodule module = nullptr;
  CUfunction prepare_inputs = nullptr;
  CUfunction composite = nullptr;
  bool live = false;

  PreparedEyeContactKernels() = default;
  PreparedEyeContactKernels(const PreparedEyeContactKernels &) = delete;
  PreparedEyeContactKernels &
  operator=(const PreparedEyeContactKernels &) = delete;
  PreparedEyeContactKernels(PreparedEyeContactKernels &&other) noexcept;
  PreparedEyeContactKernels &operator=(PreparedEyeContactKernels &&) = delete;
};

struct PreparedEyeContactInputs {
  // Exact dlib-68 order, interleaved x/y, frame-pixel coordinates.
  const CudaTensor *landmarks = nullptr; // [1,1,1,136], producer-exact
  CudaTensor *crop_geometry = nullptr;   // [1,2,1,5]: l,t,w,h,valid
  CudaTensor *left_eye_rgb = nullptr;    // [1,3,48,64]
  CudaTensor *right_eye_rgb = nullptr;   // [1,3,48,64]
  CudaTensor *left_anchors = nullptr;    // [1,12,48,64]
  CudaTensor *right_anchors = nullptr;   // [1,12,48,64]
  CudaTensor *left_angles = nullptr;     // [1,2,1,1]
  CudaTensor *right_angles = nullptr;    // [1,2,1,1]
};

bool PrepareEyeContactKernels(CudaDriverApi &driver, CUcontext owner_context,
                              CUstream owner_stream,
                              PreparedEyeContactKernels *out,
                              std::string *error_out);

// Enqueues landmark-to-crop, RGB NCHW, 12-channel anchor, and angle generation.
// No allocation, module loading, transfer, or synchronization occurs.
bool LaunchEyeContactInputsPrepared(const PreparedEyeContactKernels &kernels,
                                    const CudaImage &pre_eye_rgb,
                                    const PreparedEyeContactInputs &outputs,
                                    float strength, std::string *error_out);

// Writes a disjoint output frame. Each eye is used only when its resident crop
// flag and corresponding model-success flag are true. Invalid eyes preserve
// pre_eye_rgb exactly. Eye outputs are exact [1,3,48,64] RGB float tensors.
bool LaunchEyeContactCompositePrepared(
    const PreparedEyeContactKernels &kernels, const CudaImage &pre_eye_rgb,
    const CudaTensor &crop_geometry, const CudaTensor &left_model_rgb,
    const CudaTensor &right_model_rgb, bool left_model_succeeded,
    bool right_model_succeeded, float strength, CudaImage *output_rgb,
    std::string *error_out);

// Caller first quiesces owner_stream and makes owner_context current. Failed
// unload retains all handles so destruction can be retried.
bool DestroyEyeContactKernelsPrepared(PreparedEyeContactKernels *kernels,
                                      std::string *error_out);

} // namespace studiocast::cuda::kernels
