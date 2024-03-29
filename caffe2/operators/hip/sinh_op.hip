// !!! This is a file automatically generated by hipify!!!
#include "hip/hip_runtime.h"
#include "caffe2/operators/sinh_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_gpu.h"

namespace caffe2 {

namespace {

__global__ void SinhGradientHIPKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * coshf(__ldg(X + i));
#else
    dX[i] = dY[i] * coshf(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SinhGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& /* dY_dims */,
    const std::vector<int>& X_dims,
    const T* dY,
    const T* X,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  SinhGradientHIPKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_HIP_NUM_THREADS,
      0,
      context->hip_stream()>>>(size, dY, X, dX);
  C10_HIP_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_HIP_OPERATOR(
    Sinh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SinhFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    SinhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SinhGradientFunctor<HIPContext>>);

} // namespace caffe2
