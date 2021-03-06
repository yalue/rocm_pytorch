// !!! This is a file automatically generated by hipify!!!
#pragma once

#include <rocblas.h>
#include <hipsparse.h>

#ifdef CUDART_VERSION
#include <cusolver_common.h>
#endif

#include <ATen/Context.h>
#include <c10/util/Exception.h>
#include <c10/hip/HIPException.h>

namespace c10 {

class CuDNNError : public c10::Error {
  using Error::Error;
};

}  // namespace c10

#define AT_CUDNN_CHECK_WITH_SHAPES(EXPR, ...) AT_CUDNN_CHECK(EXPR, "\n", ##__VA_ARGS__)

// See Note [CHECK macro]
#define AT_CUDNN_CHECK(EXPR, ...)                                                               \
  do {                                                                                          \
    cudnnStatus_t status = EXPR;                                                                \
    if (status != CUDNN_STATUS_SUCCESS) {                                                       \
      if (status == CUDNN_STATUS_NOT_SUPPORTED) {                                               \
        TORCH_CHECK_WITH(CuDNNError, false,                                                     \
            "cuDNN error: ",                                                                    \
            cudnnGetErrorString(status),                                                        \
            ". This error may appear if you passed in a non-contiguous input.", ##__VA_ARGS__); \
      } else {                                                                                  \
        TORCH_CHECK_WITH(CuDNNError, false,                                                     \
            "cuDNN error: ", cudnnGetErrorString(status), ##__VA_ARGS__);                       \
      }                                                                                         \
    }                                                                                           \
  } while (0)

namespace at { namespace cuda { namespace blas {
const char* _cublasGetErrorEnum(rocblas_status error);
}}} // namespace at::cuda::blas

#define TORCH_CUDABLAS_CHECK(EXPR)                              \
  do {                                                          \
    rocblas_status __err = EXPR;                                \
    TORCH_CHECK(__err == rocblas_status_success,                 \
                "CUDA error: ",                                 \
                at::cuda::blas::_cublasGetErrorEnum(__err),     \
                " when calling `" #EXPR "`");                   \
  } while (0)

const char *cusparseGetErrorString(hipsparseStatus_t status);

#define TORCH_CUDASPARSE_CHECK(EXPR)                            \
  do {                                                          \
    hipsparseStatus_t __err = EXPR;                              \
    TORCH_CHECK(__err == HIPSPARSE_STATUS_SUCCESS,               \
                "CUDA error: ",                                 \
                cusparseGetErrorString(__err),                  \
                " when calling `" #EXPR "`");                   \
  } while (0)

// cusolver related headers are only supported on cuda now
#ifdef CUDART_VERSION

namespace at { namespace cuda { namespace solver {
const char* cusolverGetErrorMessage(cusolverStatus_t status);
}}} // namespace at::cuda::solver

#define TORCH_CUSOLVER_CHECK(EXPR)                                \
  do {                                                            \
    cusolverStatus_t __err = EXPR;                                \
    TORCH_CHECK(__err == CUSOLVER_STATUS_SUCCESS,                 \
                "cusolver error: ",                               \
                at::cuda::solver::cusolverGetErrorMessage(__err), \
                ", when calling `" #EXPR "`");                    \
  } while (0)

#else
#define TORCH_CUSOLVER_CHECK(EXPR) EXPR
#endif

#define AT_CUDA_CHECK(EXPR) C10_HIP_CHECK(EXPR)

// For CUDA Driver API
//
// This is here instead of in c10 because NVRTC is loaded dynamically via a stub
// in ATen, and we need to use its hiprtcGetErrorString.
// See NOTE [ USE OF NVRTC AND DRIVER API ].
#ifndef __HIP_PLATFORM_HCC__

#define AT_CUDA_DRIVER_CHECK(EXPR)                                                                               \
  do {                                                                                                           \
    hipError_t __err = EXPR;                                                                                       \
    if (__err != hipSuccess) {                                                                                 \
      const char* err_str;                                                                                       \
      hipError_t get_error_str_err C10_UNUSED = at::globalContext().getNVRTC().hipGetErrorString___(__err, &err_str);  \
      if (get_error_str_err != hipSuccess) {                                                                   \
        AT_ERROR("CUDA driver error: unknown error");                                                            \
      } else {                                                                                                   \
        AT_ERROR("CUDA driver error: ", err_str);                                                                \
      }                                                                                                          \
    }                                                                                                            \
  } while (0)

#else

#define AT_CUDA_DRIVER_CHECK(EXPR)                                                \
  do {                                                                            \
    hipError_t __err = EXPR;                                                        \
    if (__err != hipSuccess) {                                                  \
      AT_ERROR("CUDA driver error: ", static_cast<int>(__err));                   \
    }                                                                             \
  } while (0)

#endif

// For CUDA NVRTC
//
// Note: As of CUDA 10, nvrtc error code 7, HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE,
// incorrectly produces the error string "NVRTC unknown error."
// The following maps it correctly.
//
// This is here instead of in c10 because NVRTC is loaded dynamically via a stub
// in ATen, and we need to use its hiprtcGetErrorString.
// See NOTE [ USE OF NVRTC AND DRIVER API ].
#define AT_CUDA_NVRTC_CHECK(EXPR)                                                                   \
  do {                                                                                              \
    hiprtcResult __err = EXPR;                                                                       \
    if (__err != HIPRTC_SUCCESS) {                                                                   \
      if (static_cast<int>(__err) != 7) {                                                           \
        AT_ERROR("CUDA NVRTC error: ", at::globalContext().getNVRTC().hiprtcGetErrorString(__err));  \
      } else {                                                                                      \
        AT_ERROR("CUDA NVRTC error: HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE");                        \
      }                                                                                             \
    }                                                                                               \
  } while (0)
