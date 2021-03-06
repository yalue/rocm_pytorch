// !!! This is a file automatically generated by hipify!!!
#include "hip/hip_runtime.h"
// Based on the simpleTempltes CUDA example

#ifndef THCUNN_SHAREDMEM_H
#define THCUNN_SHAREDMEM_H

template <typename T>
struct SharedMem {
  __device__ T *getPointer()
  {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

template <>
struct SharedMem<half>
{
  __device__ half *getPointer() {
    HIP_DYNAMIC_SHARED( half, s_half)
    return s_half;
  }
};

template <>
struct SharedMem<float>
{
  __device__ float *getPointer() {
    HIP_DYNAMIC_SHARED( float, s_float)
    return s_float;
  }
};

template <>
struct SharedMem<double>
{
  __device__ double *getPointer() {
    HIP_DYNAMIC_SHARED( double, s_double)
    return s_double;
  }
};

#endif
