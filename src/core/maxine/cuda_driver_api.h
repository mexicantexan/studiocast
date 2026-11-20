#pragma once

// Compatibility forwarding surface for the protected camera pipeline and
// source consumers that still include the historical Maxine-owned path.
// New code must include core/cuda/cuda_driver_api.h and use studiocast::cuda.
#include "core/cuda/cuda_driver_api.h"

namespace studiocast::maxine {

using CUctx_st = studiocast::cuda::CUctx_st;
using CUmod_st = studiocast::cuda::CUmod_st;
using CUfunc_st = studiocast::cuda::CUfunc_st;
using CUstream_st = studiocast::cuda::CUstream_st;
using CUarray_st = studiocast::cuda::CUarray_st;

using CUcontext = studiocast::cuda::CUcontext;
using CUmodule = studiocast::cuda::CUmodule;
using CUfunction = studiocast::cuda::CUfunction;
using CUstream = studiocast::cuda::CUstream;
using CUarray = studiocast::cuda::CUarray;
using CUdeviceptr = studiocast::cuda::CUdeviceptr;
using CUdevice = studiocast::cuda::CUdevice;
using CUmemorytype = studiocast::cuda::CUmemorytype;
using CUDA_MEMCPY2D = studiocast::cuda::CUDA_MEMCPY2D;
using CUresult = studiocast::cuda::CUresult;
using CUjit_option = studiocast::cuda::CUjit_option;
using CudaDriverApi = studiocast::cuda::CudaDriverApi;

inline constexpr auto CU_MEMORYTYPE_HOST =
    studiocast::cuda::CU_MEMORYTYPE_HOST;
inline constexpr auto CU_MEMORYTYPE_DEVICE =
    studiocast::cuda::CU_MEMORYTYPE_DEVICE;
inline constexpr auto CU_MEMORYTYPE_ARRAY =
    studiocast::cuda::CU_MEMORYTYPE_ARRAY;
inline constexpr auto CU_MEMORYTYPE_UNIFIED =
    studiocast::cuda::CU_MEMORYTYPE_UNIFIED;
inline constexpr auto CUDA_SUCCESS = studiocast::cuda::CUDA_SUCCESS;
inline constexpr auto CU_JIT_INFO_LOG_BUFFER =
    studiocast::cuda::CU_JIT_INFO_LOG_BUFFER;
inline constexpr auto CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES =
    studiocast::cuda::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
inline constexpr auto CU_JIT_ERROR_LOG_BUFFER =
    studiocast::cuda::CU_JIT_ERROR_LOG_BUFFER;
inline constexpr auto CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES =
    studiocast::cuda::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
inline constexpr auto CU_JIT_LOG_VERBOSE =
    studiocast::cuda::CU_JIT_LOG_VERBOSE;

} // namespace studiocast::maxine
