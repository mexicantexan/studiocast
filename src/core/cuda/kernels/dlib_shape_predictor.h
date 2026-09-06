#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_image.h"

#include <string>

namespace studiocast::cuda::kernels {

struct PreparedDlibShapePredictorKernel {
  CudaDriverApi *driver = nullptr;
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

bool PrepareDlibShapePredictorKernel(CudaDriverApi &, CUcontext, CUstream,
                                     PreparedDlibShapePredictorKernel *,
                                     std::string *);
bool LaunchDlibShapePredictor(
    const PreparedDlibShapePredictorKernel &, const CudaImage &, float, float,
    float, float, CUdeviceptr initial_shape, CUdeviceptr cascade_tree_offsets,
    CUdeviceptr tree_split_offsets, CUdeviceptr tree_leaf_offsets,
    CUdeviceptr splits, CUdeviceptr leaves, CUdeviceptr cascade_feature_offsets,
    CUdeviceptr anchors, CUdeviceptr deltas, unsigned int cascades,
    CUdeviceptr scratch_shape, CUdeviceptr scratch_features,
    CUdeviceptr output_landmarks, std::string *);
bool DestroyDlibShapePredictorKernel(PreparedDlibShapePredictorKernel *,
                                     std::string *);

} // namespace studiocast::cuda::kernels
