// !!! This is a file automatically generated by hipify!!!
#include "hip/hip_runtime.h"
#include "caffe2/operators/tile_op.h"

#include <array>

#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void TileCopyHIPKernel(
    const int total_size,
    const int inner_size,
    const int tiles,
    const T* X,
    T* Y) {
  const int x = blockIdx.x * CAFFE_HIP_NUM_THREADS + threadIdx.x;
  if (x < total_size) {
    const int r = x / inner_size / tiles;
    const int c = x % inner_size;
#if __HIP_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[x] = __ldg(X + r * inner_size + c);
#else
    Y[x] = X[r * inner_size + c];
#endif
  }
}

} // namespace

template <>
template <typename T>
bool TileOp<HIPContext>::DoTile(
    const int outer_size,
    const int inner_size,
    const T* X,
    T* Y) {
  const std::int64_t total_size = static_cast<std::int64_t>(outer_size) *
      static_cast<std::int64_t>(tiles_) * static_cast<std::int64_t>(inner_size);
  const int M = math::DivUp<std::int64_t>(total_size, CAFFE_HIP_NUM_THREADS);
  TileCopyHIPKernel<T>
      <<<M, CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream()>>>(
          total_size, inner_size, tiles_, X, Y);
  C10_HIP_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool TileGradientOp<HIPContext>::DoTileGradient(
    const int outer_size,
    const int inner_size,
    const T* dY,
    T* dX) {
  const std::array<int, 3> dY_dims = {outer_size, tiles_, inner_size};
  const std::array<int, 3> dX_dims = {outer_size, 1, inner_size};
  math::ReduceSum<T, HIPContext>(
      3, dY_dims.data(), dX_dims.data(), T(1), dY, dX, &context_);
  return true;
}

template <>
template <>
bool TileGradientOp<HIPContext>::DoTileGradient<float>(
    const int outer_size,
    const int inner_size,
    const float* dY,
    float* dX) {
  if (inner_size == 1) {
    const std::array<int, 2> dY_dims = {outer_size, tiles_};
    const std::array<int, 2> dX_dims = {outer_size, 1};
    math::ReduceSum<float, HIPContext>(
        2, dY_dims.data(), dX_dims.data(), 1.0f, dY, dX, &context_);
  } else {
    ReinitializeTensor(&ones_, tiles_, at::dtype<float>().device(HIP));
    math::Set<float, HIPContext>(
        tiles_, 1.0f, ones_.template mutable_data<float>(), &context_);
    math::GemmStridedBatched<float, HIPContext>(
        CblasTrans,
        CblasNoTrans,
        outer_size,
        inner_size,
        1,
        tiles_,
        1.0f,
        dY,
        tiles_ * inner_size,
        ones_.template data<float>(),
        0,
        0.0f,
        dX,
        inner_size,
        &context_);
  }
  return true;
}

REGISTER_HIP_OPERATOR(Tile, TileOp<HIPContext>);
REGISTER_HIP_OPERATOR(TileGradient, TileGradientOp<HIPContext>);

} // namespace caffe2
