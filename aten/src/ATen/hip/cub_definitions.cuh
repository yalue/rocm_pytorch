// !!! This is a file automatically generated by hipify!!!
#pragma once

#if !defined(USE_ROCM)
#include <hip/hip_runtime.h>  // for TORCH_HIP_VERSION
#endif

#if defined(TORCH_HIP_VERSION) && TORCH_HIP_VERSION >= 11000
#include <cub/version.cuh>
#else
#define CUB_VERSION 0
#endif

// cub sort support for __nv_bfloat16 is added to cub 1.13 in:
// https://github.com/NVIDIA/cub/pull/306
#if CUB_VERSION >= 101300
#define CUB_SUPPORTS_NV_BFLOAT16() true
#else
#define CUB_SUPPORTS_NV_BFLOAT16() false
#endif

// cub sort support for CUB_WRAPPED_NAMESPACE is added to cub 1.13.1 in:
// https://github.com/NVIDIA/cub/pull/326
// CUB_WRAPPED_NAMESPACE is defined globally in cmake/Dependencies.cmake
// starting from CUDA 11.5
#if defined(CUB_WRAPPED_NAMESPACE) || defined(THRUST_CUB_WRAPPED_NAMESPACE)
#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() true
#else
#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() false
#endif